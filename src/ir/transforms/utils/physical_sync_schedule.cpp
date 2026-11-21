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
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/core_affinity.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/scoped_sync_palette.h"
#include "src/ir/transforms/utils/scoped_token_families.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_placement.h"
#include "src/ir/transforms/utils/sync_token_names.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace physical {
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
  const ForStmt* enclosing_loop_ = nullptr;
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
    auto* saved_loop = enclosing_loop_;
    enclosing_loop_ = loop.get();
    VisitStmt(loop->body_);
    enclosing_loop_ = saved_loop;
    if (inserted) scope_.ready.erase(loop->loop_var_.get());
    uniform_variables_ = std::move(saved_variables);
    domain_ = std::move(saved);
  }
  void VisitStmt_(const AssignStmtPtr& statement) override { QualifyCut(As<Call>(statement->value_)); }
  void VisitStmt_(const IfStmtPtr& branch) override {
    ++conditional_depth_;
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    --conditional_depth_;
  }
  void VisitStmt_(const EvalStmtPtr& statement) override { QualifyCut(As<Call>(statement->expr_)); }
  void QualifyCut(const CallPtr& call) {
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
        if (!conditional_depth_) cut.enclosing_loop = enclosing_loop_;
      } else {
        if (conditional_depth_ || cut.enclosing_loop != enclosing_loop_) cut.enclosing_loop = nullptr;
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
             std::pair<int64_t, int64_t> blocks, PhysicalSyncRealization& work, ExprPtr coordinate = nullptr)
      : absent_(absent), cuts_(cuts), domain_{blocks}, work_(work), scope_{owner, nullptr, {}} {
    auto entry = std::make_shared<std::set<const Var*>>();
    for (const auto& parameter : owner->params_) {
      const auto type = As<ScalarType>(parameter->GetType());
      if (type && (type->dtype_ == DataType::INDEX || type->dtype_ == DataType::BOOL))
        entry->insert(parameter.get());
    }
    // The retained invocation coordinate (or original leading launch lookup)
    // dominates every payload. Other scalar assignments confer no evidence.
    if (auto block = As<Var>(coordinate ? coordinate : DominatingSyncBlock(owner)))
      entry->insert(block.get());
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
  const AssignedScopedPalette* palette = nullptr;
  const ScopedPublicationRankPiece* rank = nullptr;
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
                     retained->ranges.size() == frames.size() &&
                     (retained->steps.empty() || retained->steps.size() == frames.size()))
          << "Foreign endpoint domain";
      for (size_t i = 0; i < frames.size(); ++i) {
        const auto [first, end] = retained->ranges[i];
        const auto stride = retained->steps.empty() ? 1 : retained->steps[i];
        INTERNAL_CHECK(first >= 0 && first < end && end <= frames[i].count && stride > 0 &&
                       (end - first - 1) % stride == 0)
            << "Invalid retained endpoint progression";
        INTERNAL_CHECK(context.loops[i].loop == frames[i].loop && context.loops[i].count == frames[i].count &&
                       context.loops[i].start == frames[i].start && context.loops[i].step == frames[i].step &&
                       context.loops[i].start_expression == frames[i].start_expression)
            << "Incompatible original endpoint frame";
      }
      return retained;
    }
    CHECK(std::all_of(f.axes.begin(), f.axes.end(), [](const ScopedDemandAxis& axis) {
      return axis.source_step == 1 && axis.target_step == 1;
    })) << "Periodic scoped endpoints require retained exact activity domains";
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
  std::shared_ptr<GuardDomainArena> arena = std::make_shared<GuardDomainArena>();
  std::vector<AssignedCut> cuts;
  std::map<const ForStmt*, LoopBoundaryDomain> loop_boundaries;
  AssignedRegionSchedule(FunctionPtr function, std::vector<AssignedSyncCase> supplied)
      : owner(std::move(function)),
        block(!supplied.empty() && supplied.front().input && supplied.front().input->invocation_coordinate
                  ? supplied.front().input->invocation_coordinate->Expression()
                  : DominatingSyncBlock(owner)),
        cases(std::move(supplied)) {
    for (size_t c = 0; c < cases.size(); ++c) {
      CHECK(cases[c].input && cases[c].assignment && cases[c].work) << "Missing assigned schedule case";
      const auto& input = *cases[c].input;
      auto& work = *cases[c].work;
      CHECK(work.assignment == cases[c].assignment) << "Assigned schedule accounting belongs to another plan";
      cases[c].assignment->ValidateFor(input, input.hardware_target);
      const auto& families = cases[c].assignment->Families();
      INTERNAL_CHECK(input.owner == owner) << "Foreign assigned schedule owner";
      CHECK(input.invocation_coordinate == cases.front().input->invocation_coordinate)
          << "Assigned cases disagree on their invocation coordinate owner";
      if (cases.size() > 1 && input.block_bounds) {
        for (const auto& segment : input.segments) {
          // CutDomains uses the innermost lexical loop. A payload in an
          // outer loop's own body has a segment whose frame path ends there.
          if (segment.nested_frames.empty()) continue;
          const auto& frame = segment.nested_frames.back();
          if (!frame.count || !As<Var>(frame.start_expression) ||
              frame.start_expression != frame.loop->start_)
            continue;
          auto& domain = loop_boundaries[frame.loop.get()];
          domain.loop = frame.loop;
          if (domain.cases.empty() || domain.cases.back().blocks != *input.block_bounds) {
            domain.cases.push_back({*input.block_bounds, frame.count});
          } else {
            INTERNAL_CHECK_SPAN(domain.cases.back().count == frame.count, frame.loop->span_)
                << "Inconsistent trip count for the same original loop and launch case";
          }
        }
      }
      ScheduleDomains domains(input);
      if (const auto& protocol = cases[c].assignment->Protocol()) {
        protocol->ValidateFor(input);
        const auto& mapping = cases[c].assignment->ProtocolAssignments();
        for (const auto& original : protocol->Cuts()) {
          const auto& payload = input.segments[original.region].sites[original.site];
          AssignedCut cut{c, payload.original, payload.continuation, {}, {}};
          std::set<size_t> seen_before, seen_after;
          auto append = [&](size_t raw, bool publish, std::vector<AssignedEndpoint>& endpoints,
                            std::set<size_t>& seen) {
            const auto& use = protocol->Families()[raw];
            const auto assigned = mapping.at(raw);
            const auto& raw_palettes = cases[c].assignment->RawPalettes();
            if (!raw_palettes.empty() && raw_palettes[raw].palette) {
              const auto& palette = cases[c].assignment->Palettes().at(*raw_palettes[raw].palette);
              const auto& pieces = palette.certificate->Schedule()->Pieces();
              for (auto index : raw_palettes[raw].pieces) {
                const auto& piece = pieces.at(index);
                INTERNAL_CHECK(piece.raw_use == raw) << "Assigned rank piece has a foreign raw use";
                const auto& digits = publish ? piece.source : piece.target;
                auto domain = std::make_shared<SyncEndpointDomain>();
                domain->frame = domains.Get(use, publish)->frame;
                for (const auto& digit : digits) {
                  const auto end = static_cast<__int128>(digit.first) +
                                   static_cast<__int128>(digit.count - 1) * digit.step + 1;
                  INTERNAL_CHECK(end <= std::numeric_limits<int64_t>::max())
                      << "Rank piece endpoint overflow";
                  domain->ranges.emplace_back(digit.first, static_cast<int64_t>(end));
                  domain->steps.push_back(digit.step);
                }
                endpoints.push_back({c, assigned, publish, std::move(domain), &palette, &piece});
                ++work.assigned_schedule_endpoints;
              }
            } else if (seen.insert(assigned).second) {
              INTERNAL_CHECK(families[assigned].ids.size() <= 1)
                  << "Protocol view requires a certified raw-record physical rank map";
              endpoints.push_back({c, assigned, publish, domains.Get(families[assigned], publish)});
              ++work.assigned_schedule_endpoints;
            }
          };
          for (auto raw : original.before) append(raw, false, cut.before, seen_before);
          for (auto raw : original.after) append(raw, true, cut.after, seen_after);
          cuts.push_back(std::move(cut));
        }
        continue;
      }
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
  while (auto seq = As<SeqStmts>(body)) {
    size_t end = seq->stmts_.size();
    if (end > 0) {
      if (const auto yield = As<YieldStmt>(seq->stmts_[end - 1]); yield && yield->value_.empty()) {
        --end;
      }
    }
    if (end == 0) return nullptr;
    for (size_t i = 0; i + 1 < end; ++i) {
      const auto assign = As<AssignStmt>(seq->stmts_[i]);
      if (!assign) return nullptr;
      if (!As<ScalarType>(assign->var_->GetType()) && !IsOp(As<Call>(assign->value_), "buffer.slot")) {
        return nullptr;
      }
    }
    body = seq->stmts_[end - 1];
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
          // Interval-only phase scheduling has no residue proof for these
          // domains. Keep the exact guarded path instead of widening holes.
          if (std::any_of(endpoint.domain->steps.begin(), endpoint.domain->steps.end(),
                          [](int64_t stride) { return stride != 1; }))
            continue;
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
      const auto p =
          static_cast<int64_t>(endpoint.palette ? endpoint.palette->ids.size() : family.ids.size());
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
  struct ScopedDomainEntry {
    bool active = false;
    RadixBoxGuard guard = RadixBoxGuard::True();
    std::vector<AxisId> axes;
  };
  std::map<const SyncEndpointDomain*, ScopedDomainEntry> scoped_domains_;

  static bool FitsOriginalBoundary(const SyncLoopFrame& frame, int64_t digit) {
    return OriginalLoopBoundary(frame.loop, frame.start, frame.step, frame.count, frame.start_expression,
                                digit)
        .has_value();
  }

  ScopedDomainEntry ResolveScopedDomain(const SyncEndpointDomain& domain) {
    auto it = scoped_domains_.find(&domain);
    if (it != scoped_domains_.end()) return it->second;
    ScopedDomainEntry entry;
    const auto& frames = domain.frame->loops;
    result_.endpoint_domain_entries += domain.ranges.size();
    RadixBox base_box;
    entry.axes.reserve(frames.size());
    for (size_t i = 0; i < frames.size(); ++i) {
      const AxisId ax = schedule_.arena->RegisterLoopFrame(frames[i]);
      entry.axes.push_back(ax);
      const auto [raw_first, raw_end] = domain.ranges[i];
      INTERNAL_CHECK(raw_first >= 0 && raw_end > raw_first && raw_end <= frames[i].count)
          << "Invalid endpoint interval";
      const int64_t raw_stride = domain.steps.empty() ? 1 : domain.steps[i];
      auto ap = AxisAP::MakeCanonical(raw_first, raw_end, raw_stride, frames[i].count);
      if (!ap) return scoped_domains_.emplace(&domain, std::move(entry)).first->second;
      if (phase_ && frames[i].loop == phase_->loop) {
        const auto phase_ap = AxisAP::MakeCanonical(phase_->first, phase_->end, 1, frames[i].count);
        if (!phase_ap) return scoped_domains_.emplace(&domain, std::move(entry)).first->second;
        ap = CRTIntersect(*ap, *phase_ap, frames[i].count);
        if (!ap) return scoped_domains_.emplace(&domain, std::move(entry)).first->second;
        if (phase_->first >= raw_first && phase_->end <= raw_end && ap->stride == 1) continue;
      }
      if (!ap->IsFull(frames[i].count)) {
        if (ap->IsSingleton()) {
          ++result_.endpoint_predicates_constructed;
          if (FitsOriginalBoundary(frames[i], ap->first)) ++result_.normalized_endpoint_bounds;
        } else {
          const bool need_mod = ap->stride > 1;
          const bool need_ge = ap->first - ap->stride >= 0;
          const bool need_lt = ap->Last() + ap->stride < frames[i].count;
          if (need_mod) ++result_.endpoint_predicates_constructed;
          if (need_ge) ++result_.endpoint_predicates_constructed;
          if (need_lt) ++result_.endpoint_predicates_constructed;
          if ((need_ge || need_lt) && (!need_ge || FitsOriginalBoundary(frames[i], ap->first)) &&
              (!need_lt || FitsOriginalBoundary(frames[i], ap->Last() + 1))) {
            if (need_ge) ++result_.normalized_endpoint_bounds;
            if (need_lt) ++result_.normalized_endpoint_bounds;
          }
        }
        base_box.constraints.push_back(AxisConstraint{ax, *ap});
      }
    }
    entry.active = true;
    entry.guard = RadixBoxGuard::FromBox(schedule_.arena, std::move(base_box));
    return scoped_domains_.emplace(&domain, std::move(entry)).first->second;
  }

  static void DecomposeMixedRadix(int64_t lo, int64_t hi, size_t depth, int64_t total,
                                  const std::vector<AxisId>& axes, const std::vector<SyncLoopFrame>& frames,
                                  const RadixBox& prefix, std::vector<RadixBox>& out, bool& overflow) {
    if (overflow || lo >= hi) return;
    if (lo == 0 && hi == total) {
      out.push_back(prefix);
      if (out.size() > RadixBoxGuard::kMaxGuardBoxes) overflow = true;
      return;
    }
    const int64_t stride = total / frames[depth].count;
    int64_t d_lo = lo / stride;
    const int64_t r_lo = lo % stride;
    int64_t d_hi = (hi - 1) / stride;
    const int64_t r_hi = (hi - 1) % stride + 1;
    auto push_axis = [&](RadixBox box, int64_t first, int64_t end) {
      if (const auto ap = AxisAP::MakeCanonical(first, end, 1, frames[depth].count)) {
        if (!ap->IsFull(frames[depth].count)) {
          box.constraints.push_back(AxisConstraint{axes[depth], *ap});
        }
      }
      return box;
    };
    if (d_lo == d_hi) {
      DecomposeMixedRadix(r_lo, r_hi, depth + 1, stride, axes, frames, push_axis(prefix, d_lo, d_lo + 1), out,
                          overflow);
      return;
    }
    if (r_lo > 0) {
      DecomposeMixedRadix(r_lo, stride, depth + 1, stride, axes, frames, push_axis(prefix, d_lo, d_lo + 1),
                          out, overflow);
      ++d_lo;
    }
    int64_t d_mid_end = d_hi;
    if (r_hi < stride) {
      --d_mid_end;
    }
    if (d_lo <= d_mid_end) {
      out.push_back(push_axis(prefix, d_lo, d_mid_end + 1));
      if (out.size() > RadixBoxGuard::kMaxGuardBoxes) overflow = true;
    }
    if (r_hi < stride) {
      DecomposeMixedRadix(0, r_hi, depth + 1, stride, axes, frames, push_axis(prefix, d_hi, d_hi + 1), out,
                          overflow);
    }
  }

  RadixBoxGuard BuildPhaseScope() const {
    if (!phase_ || !phase_->loop) return RadixBoxGuard::True();
    for (const auto& seg : input_.segments) {
      for (const auto& frame : seg.nested_frames) {
        if (frame.loop == phase_->loop) {
          const AxisId ax = schedule_.arena->RegisterLoopFrame(frame);
          const auto ap = AxisAP::MakeCanonical(phase_->first, phase_->end, 1, frame.count);
          if (!ap) return RadixBoxGuard::False();
          if (ap->IsFull(frame.count)) return RadixBoxGuard::True();
          RadixBox box;
          box.constraints.push_back(AxisConstraint{ax, *ap});
          return RadixBoxGuard::FromBox(schedule_.arena, std::move(box));
        }
      }
      if (seg.original_loop == phase_->loop) {
        const SyncLoopFrame frame{seg.original_loop, seg.count, seg.start, seg.step, nullptr};
        const AxisId ax = schedule_.arena->RegisterLoopFrame(frame);
        const auto ap = AxisAP::MakeCanonical(phase_->first, phase_->end, 1, frame.count);
        if (!ap) return RadixBoxGuard::False();
        if (ap->IsFull(frame.count)) return RadixBoxGuard::True();
        RadixBox box;
        box.constraints.push_back(AxisConstraint{ax, *ap});
        return RadixBoxGuard::FromBox(schedule_.arena, std::move(box));
      }
    }
    return RadixBoxGuard::True();
  }

  ControlPtr LowerAssignedCommand(const AssignedEndpoint& endpoint, const Span& span) {
    INTERNAL_CHECK(endpoint.case_index == case_) << "Foreign assigned endpoint case";
    const auto& f = schedule_.Record(endpoint);
    const bool publish = endpoint.publish;
    const auto [src, dst] = SyncFamilyPipes(input_, f);
    const size_t r = publish ? f.source_region : f.target_region;
    const int64_t first = publish ? f.source_first : f.target_first;
    const bool scoped = input_.region_tree.has_value();
    const auto& ids = endpoint.palette ? endpoint.palette->ids : f.ids;
    INTERNAL_CHECK(!scoped || endpoint.palette || ids.size() <= 1)
        << "Internal error: unsupported scoped physical period";

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

    std::vector<AxisId> region_axes;
    RadixBoxGuard base_guard = RadixBoxGuard::True();
    if (scoped) {
      const auto resolved = ResolveScopedDomain(*endpoint.domain);
      if (!resolved.active) return ControlSequence({}, span, result_);
      region_axes = resolved.axes;
      base_guard = resolved.guard;
    } else {
      const auto& seg = input_.segments[r];
      if (!seg.original_loop && seg.nested_frames.empty()) {
        base_guard = RadixBoxGuard::True();
      } else if (seg.nested_frames.empty()) {
        const SyncLoopFrame frame{seg.original_loop, seg.count, seg.start, seg.step, nullptr};
        const AxisId ax = schedule_.arena->RegisterLoopFrame(frame);
        region_axes.push_back(ax);
        int64_t lo = first;
        int64_t hi = first + f.count;
        if (phase_ && seg.original_loop == phase_->loop) {
          lo = std::max(lo, phase_->first);
          hi = std::min(hi, phase_->end);
        }
        const auto ap = AxisAP::MakeCanonical(lo, hi, 1, seg.count);
        if (!ap) return ControlSequence({}, span, result_);
        RadixBox base_box;
        if (!ap->IsFull(seg.count) && !(phase_ && seg.original_loop == phase_->loop &&
                                        phase_->first >= first && phase_->end <= first + f.count)) {
          base_box.constraints.push_back(AxisConstraint{ax, *ap});
        }
        base_guard = RadixBoxGuard::FromBox(schedule_.arena, std::move(base_box));
      } else {
        for (const auto& frame : seg.nested_frames) {
          region_axes.push_back(schedule_.arena->RegisterLoopFrame(frame));
        }
        if (first > 0 || first + f.count < seg.count) {
          std::vector<RadixBox> boxes;
          bool overflow = false;
          DecomposeMixedRadix(first, first + f.count, 0, seg.count, region_axes, seg.nested_frames,
                              RadixBox{}, boxes, overflow);
          CHECK(!overflow) << "Flat mixed-radix presence range exceeded box budget";
          base_guard = RadixBoxGuard::FromBoxes(schedule_.arena, std::move(boxes));
        }
      }
    }

    ControlForest commands;
    const std::string core = input_.owner->func_type_ == FunctionType::AIV ? "aiv" : "aic";
    if (src == dst) {
      const auto* name = RequireLocalSyncBarrier(dst, input_.owner->func_type_);
      INTERNAL_CHECK(name) << "Missing qualified same-pipe barrier";
      auto call = OpRegistry::GetInstance().Create(name, {}, {{"core_type", core}}, span);
      commands.push_back(ControlCommand(call, span, result_));
      ++result_.barrier_sites;
    } else {
      const bool literal_residue = phase_ && phase_->residue.has_value() && !scoped;
      if (ids.size() <= 1 || literal_residue) {
        size_t idx = 0;
        if (literal_residue && ids.size() > 1) {
          const int64_t period = static_cast<int64_t>(ids.size());
          const int64_t digit =
              static_cast<int64_t>((__int128(phase_->first) + *phase_->residue - first) % period);
          idx = static_cast<size_t>((digit + period) % period);
        }
        auto call = OpRegistry::GetInstance().Create(publish ? "system.set_flag" : "system.wait_flag", {},
                                                     {{"src_pipe", static_cast<int>(src)},
                                                      {"dst_pipe", static_cast<int>(dst)},
                                                      {"event_id", ids[idx]},
                                                      {"core_type", core}},
                                                     span);
        commands.push_back(ControlCommand(call, span, result_));
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      } else {
        const int64_t modulus = static_cast<int64_t>(ids.size());
        RankCongruence template_rc;
        template_rc.modulus = modulus;
        if (endpoint.palette) {
          INTERNAL_CHECK(endpoint.rank && endpoint.domain) << "Missing certified publication rank";
          INTERNAL_CHECK(static_cast<__int128>(modulus) * (modulus - 1) <=
                         std::numeric_limits<int64_t>::max())
              << "Directional palette is too large for bounded modular INDEX arithmetic";
          template_rc.base = PositiveMod64(endpoint.rank->rank_base, modulus);
          const auto& src_digits = endpoint.rank->source;
          const auto& digits = publish ? endpoint.rank->source : endpoint.rank->target;
          for (size_t i = 0; i < endpoint.rank->rank_coefficients.size(); ++i) {
            const int64_t coefficient = PositiveMod64(endpoint.rank->rank_coefficients[i], modulus);
            INTERNAL_CHECK(i < src_digits.size()) << "Rank coefficient out of source bounds";
            if (!coefficient || src_digits[i].count == 1) continue;
            INTERNAL_CHECK(i < digits.size() && i < endpoint.domain->frame->loops.size() &&
                           digits[i].count == src_digits[i].count)
                << "Matched WAIT rank lost original source coordinate";
            template_rc.terms.push_back(
                RankTerm{region_axes[i], digits[i].first, digits[i].step, coefficient});
          }
        } else {
          template_rc.base = PositiveMod64(-first, modulus);
          const auto& seg = input_.segments[r];
          if (seg.nested_frames.empty()) {
            if (seg.original_loop) template_rc.terms.push_back(RankTerm{region_axes[0], 0, 1, 1 % modulus});
          } else {
            int64_t stride = 1;
            for (size_t i = seg.nested_frames.size(); i-- > 0;) {
              const int64_t coeff = stride % modulus;
              if (coeff != 0 && seg.nested_frames[i].count > 1) {
                template_rc.terms.push_back(RankTerm{region_axes[i], 0, 1, coeff});
              }
              stride = (stride * (seg.nested_frames[i].count % modulus)) % modulus;
            }
          }
        }

        struct Arm {
          RadixBoxGuard guard;
          ControlPtr cmd;
        };
        std::vector<Arm> arms;
        for (size_t i = 0; i < ids.size(); ++i) {
          RankCongruence arm_rc = template_rc;
          arm_rc.residues = ResidueMask(modulus);
          arm_rc.residues.Set(static_cast<int64_t>(i));
          RadixBox rc_box;
          rc_box.ranks.push_back(std::move(arm_rc));
          const RadixBoxGuard arm_guard =
              IntersectGuards(base_guard, RadixBoxGuard::FromBox(schedule_.arena, std::move(rc_box)));
          if (arm_guard.IsEmpty()) continue;
          auto call = OpRegistry::GetInstance().Create(publish ? "system.set_flag" : "system.wait_flag", {},
                                                       {{"src_pipe", static_cast<int>(src)},
                                                        {"dst_pipe", static_cast<int>(dst)},
                                                        {"event_id", ids[i]},
                                                        {"core_type", core}},
                                                       span);
          arms.push_back({SimplifyGuardGiven(arm_guard, base_guard), ControlCommand(call, span, result_)});
          if (publish)
            ++result_.set_sites;
          else
            ++result_.wait_sites;
        }
        if (!arms.empty()) {
          ControlPtr selection = arms.back().cmd;
          for (size_t i = arms.size() - 1; i-- > 0;) {
            selection = ControlBranch(arms[i].guard, arms[i].cmd, selection, span, result_);
          }
          commands.push_back(selection);
        }
      }
    }

    ControlPtr body = ControlSequence(commands, span, result_);
    if (!base_guard.IsTautology()) {
      body = ControlBranch(base_guard, body, std::nullopt, span, result_);
    }
    return body;
  }

  PhysicalCuts LowerCut(const AssignedCut& cut) {
    INTERNAL_CHECK(cut.case_index == case_) << "Foreign assigned cut";
    PhysicalCuts cuts;
    const auto& span = cut.original->span_;
    for (const auto& endpoint : cut.before) cuts.before.push_back(LowerAssignedCommand(endpoint, span));
    for (const auto& endpoint : cut.after) cuts.after.push_back(LowerAssignedCommand(endpoint, span));
    const RadixBoxGuard scope = BuildPhaseScope();
    result_.guard_sites += CaseCodeSize(cuts.before, scope).guards + CaseCodeSize(cuts.after, scope).guards;
    cuts.before = ShareCommandGuards(cuts.before, result_, true, false, scope);
    cuts.after = ShareCommandGuards(cuts.after, result_, true, false, scope);
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
  for (const auto& sc : schedule.cuts) {
    const Call* anchor = sc.original.get();
    const auto it = cuts.find(anchor);
    if (it == cuts.end() || !seen.insert(it->second.get()).second) continue;
    const auto& cut = it->second;
    const auto& facts = *domains.at(anchor);
    INTERNAL_CHECK(facts.predicate_scope && facts.predicate_scope->owner == schedule.owner)
        << "Missing original-cut predicate qualification";
    cut->predicate_scope = facts.predicate_scope;
    for (const auto& item : cut->before_cases) RequireCommandScope({item.body}, *cut->predicate_scope, work);
    for (const auto& item : cut->after_cases) RequireCommandScope({item.body}, *cut->predicate_scope, work);
    const auto loop = schedule.loop_boundaries.find(facts.enclosing_loop);
    const auto* boundary_domain = !phase && loop != schedule.loop_boundaries.end() ? &loop->second : nullptr;
    cut->before = ScheduleCommandCases(block, facts.domain, cut->before_cases, work, schedule.owner->span_,
                                       facts.uniform_variables, schedule.arena, boundary_domain);
    cut->after = ScheduleCommandCases(block, facts.domain, cut->after_cases, work, schedule.owner->span_,
                                      facts.uniform_variables, schedule.arena, boundary_domain);
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
ControlForest RestrictForestAxis(const ControlForest& forest, AxisId axis, const AxisAP& range,
                                 const std::shared_ptr<const GuardDomainArena>& arena,
                                 PhysicalSyncRealization& work) {
  ControlForest out;
  const Var* axis_var = arena->Axis(axis).var.get();
  for (const auto& node : forest) {
    ++work.schedule_descriptor_visits;
    if (node->command) {
      out.push_back(node);
      continue;
    }
    if (node->guard.IsTautology()) {
      auto body = RestrictForestAxis(node->body, axis, range, arena, work);
      out.insert(out.end(), body.begin(), body.end());
      continue;
    }
    if (node->guard.IsOpaque()) {
      const bool uses_axis = std::find(node->predicate_inputs.begin(), node->predicate_inputs.end(),
                                       axis_var) != node->predicate_inputs.end();
      if (uses_axis) {
        out.push_back(node);
        continue;
      }
      auto body = RestrictForestAxis(node->body, axis, range, arena, work);
      auto alt = RestrictForestAxis(node->alternative, axis, range, arena, work);
      if (body.empty() && alt.empty()) continue;
      out.push_back(RebuildControl(node, node->guard, nullptr, std::move(body), std::move(alt), work));
      continue;
    }
    auto restricted = RestrictGuardAxis(node->guard, axis, range, arena);
    if (restricted.IsEmpty()) {
      auto alt = RestrictForestAxis(node->alternative, axis, range, arena, work);
      out.insert(out.end(), alt.begin(), alt.end());
      continue;
    }
    if (restricted.IsTautology()) {
      auto body = RestrictForestAxis(node->body, axis, range, arena, work);
      out.insert(out.end(), body.begin(), body.end());
      continue;
    }
    auto body = RestrictForestAxis(node->body, axis, range, arena, work);
    auto alt = RestrictForestAxis(node->alternative, axis, range, arena, work);
    if (body.empty() && alt.empty()) continue;
    out.push_back(
        RebuildControl(node, std::move(restricted), nullptr, std::move(body), std::move(alt), work));
  }
  return out;
}

bool ForestUsesVar(const ControlForest& forest, const Var* var) {
  for (const auto& node : forest) {
    if (std::find(node->predicate_inputs.begin(), node->predicate_inputs.end(), var) !=
        node->predicate_inputs.end())
      return true;
    if (ForestUsesVar(node->body, var) || ForestUsesVar(node->alternative, var)) return true;
  }
  return false;
}

std::shared_ptr<const GuardDomainArena> FindArenaInForest(const ControlForest& forest) {
  for (const auto& node : forest) {
    if (node->guard.Arena()) return node->guard.Arena();
    if (auto a = FindArenaInForest(node->body)) return a;
    if (auto a = FindArenaInForest(node->alternative)) return a;
  }
  return nullptr;
}

// Scan only a leading/trailing command sequence. A command active on the
// boundary is a movement fence unless it is itself extracted. Commands proved
// absent there are retained in place and do not fence other boundary commands.
ControlForest ExtractBoundary(ControlForest& original, AxisId axis, const AxisAP& boundary_ap,
                              const AxisAP& rest_ap, bool reverse,
                              const std::shared_ptr<const GuardDomainArena>& arena,
                              PhysicalSyncRealization& work) {
  ControlForest flat;
  for (const auto& node : original) FlattenControlSequence(node, flat, work);
  if (reverse) std::reverse(flat.begin(), flat.end());
  ControlForest retained, moved;
  std::vector<ControlForest> groups;
  const Var* axis_var = arena->Axis(axis).var.get();
  bool fence = false;
  for (const auto& node : flat) {
    if (!fence) {
      auto active = RestrictForestAxis({node}, axis, boundary_ap, arena, work);
      if (active.empty()) {
        retained.push_back(node);
        continue;
      }
      if (RestrictForestAxis({node}, axis, rest_ap, arena, work).empty() &&
          !ForestUsesVar(active, axis_var)) {
        groups.push_back(std::move(active));
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
                                      PhysicalSyncRealization& work,
                                      std::shared_ptr<GuardDomainArena> arena = nullptr) {
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
  std::shared_ptr<const GuardDomainArena> use_arena = arena;
  if (!use_arena) use_arena = FindArenaInForest(first->before);
  if (!use_arena) use_arena = FindArenaInForest(final->after);
  if (!arena && !use_arena) arena = std::make_shared<GuardDomainArena>();
  // Register loop axis if not already present (note: const_cast avoided by using mutable arena when
  // supplied).
  std::shared_ptr<GuardDomainArena> reg_arena = arena;
  if (!reg_arena) reg_arena = std::make_shared<GuardDomainArena>();
  const int64_t cnt = static_cast<int64_t>(count);
  const AxisId axis =
      reg_arena->RegisterLoopFrame(SyncLoopFrame{loop, cnt, start->value_, step->value_, nullptr});
  // If cuts came from another arena that already contains this loop axis, find its AxisId there.
  std::shared_ptr<const GuardDomainArena> slice_arena = reg_arena;
  AxisId slice_axis = axis;
  if (use_arena && use_arena != reg_arena) {
    bool found_axis = false;
    for (AxisId id = 0; id < use_arena->NumAxes(); ++id) {
      const auto& info = use_arena->Axis(id);
      if (info.kind == AxisKind::LoopFrame && info.var == loop->loop_var_ && info.count == cnt &&
          info.start == start->value_ && info.step == step->value_ && !info.start_expression) {
        slice_arena = use_arena;
        slice_axis = id;
        found_axis = true;
        break;
      }
    }
    if (!found_axis) return {};
  }
  const AxisAP at_first_ap{0, 1, 1};
  const AxisAP after_first_ap{1, cnt, 1};
  const AxisAP at_last_ap{cnt - 1, cnt, 1};
  const AxisAP before_last_ap{0, cnt - 1, 1};
  plan.entry = ExtractBoundary(plan.cuts[first].first, slice_axis, at_first_ap, after_first_ap, false,
                               slice_arena, work);
  plan.exit = ExtractBoundary(plan.cuts[final].second, slice_axis, at_last_ap, before_last_ap, true,
                              slice_arena, work);
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
  if (!node->guard.IsTautology()) mix(HashRadixBoxGuard(node->guard));
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
    if (!inclusive && !As<Lt>(expression) && !As<Ge>(expression) && !As<Eq>(expression) &&
        !As<Ne>(expression))
      return {};
    auto split = Split(__int128(bound->value_) + inclusive);
    if (!split) return {};
    Intervals yes;
    const auto size = static_cast<int64_t>(splits_.size() - 1);
    if (As<Eq>(expression) || As<Ne>(expression)) {
      const auto end = Split(__int128(bound->value_) + 1);
      if (!end) return {};
      if (*split < *end) yes.emplace_back(*split, *end - 1);
      if (As<Ne>(expression)) yes = SubtractIntervals(all_, yes);
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
    const std::vector<std::pair<int64_t, int64_t>>& domains, PhysicalSyncRealization& work,
    const std::vector<const Call*>& anchor_order) {
  if (domains.size() < 2) return {};
  if (!block) return {};
  std::vector<std::vector<ControlPtr>> signatures(domains.size());
  std::map<const CommandControl*, size_t> hashes;
  for (const auto& phase : phases)
    for (const auto& inventory : phase.visits) {
      std::set<const PhysicalCuts*> seen;
      auto append_cut = [&](const std::shared_ptr<PhysicalCuts>& cut) {
        if (!seen.insert(cut.get()).second) return;
        for (size_t c = 0; c < domains.size(); ++c) {
          const auto domain = domains[c];
          const auto before_leaf = CaseLeaf(cut->before_cases, domain);
          const auto after_leaf = CaseLeaf(cut->after_cases, domain);
          signatures[c].push_back(before_leaf);
          signatures[c].push_back(after_leaf);
        }
      };
      if (!anchor_order.empty()) {
        for (const Call* anchor : anchor_order) {
          if (const auto it = inventory.find(anchor); it != inventory.end()) {
            append_cut(it->second);
          }
        }
      }
      for (const auto& [anchor, cut] : inventory) append_cut(cut);
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
constexpr auto kAutoBoundaryMode = "auto_boundary";
constexpr unsigned kBoundaryMode = 2;
constexpr int64_t kMinAutoBoundaryTrips = 4;
constexpr size_t kMaxAutoBoundaryPhases = 3;

bool IsAutomaticSchedule(const SyncSchedulePolicy& policy) {
  return policy.mode == "auto" || policy.mode == kAutoBoundaryMode;
}

bool RequestsScheduleMode(const SyncSchedulePolicy& policy, unsigned mode) {
  if (policy.mode == kAutoBoundaryMode) {
    return mode == kBoundaryMode;
  }
  return policy.mode == "auto" || policy.mode == kScheduleModes[mode];
}
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
                           const std::shared_ptr<GuardDomainArena>& arena, PhysicalSyncRealization& work) {
  if ((phase.end - phase.first) / phase.period < 2) return;
  BoundaryInput first_input(phase.visits.front(), work), last_input(phase.visits.back(), work);
  first_input.Visit(loop.loop->body_);
  last_input.Visit(loop.loop->body_);
  if (!first_input.valid || !last_input.valid || first_input.cuts.empty() || last_input.cuts.empty()) return;
  const auto* a = first_input.cuts.front();
  const auto* b = last_input.cuts.back();
  auto before = a->before, after = b->after;
  const AxisId axis =
      arena->RegisterLoopFrame(SyncLoopFrame{loop.loop, loop.count, loop.start, loop.step, nullptr});
  const AxisAP at_first_ap{phase.first, phase.first + 1, 1};
  const auto rest_first_ap = AxisAP::MakeCanonical(phase.first + phase.period, phase.end - phase.period + 1,
                                                   phase.period, loop.count);
  const AxisAP at_last_ap{phase.end - 1, phase.end, 1};
  const auto rest_last_ap =
      AxisAP::MakeCanonical(phase.first + phase.period - 1, phase.end - 1, phase.period, loop.count);
  if (!rest_first_ap || !rest_last_ap) return;
  auto entry = ExtractBoundary(before, axis, at_first_ap, *rest_first_ap, false, arena, work);
  auto exit = ExtractBoundary(after, axis, at_last_ap, *rest_last_ap, true, arena, work);
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
    auto shared = ShareCommandGuards(joined, work_, false, true);
    const CaseCodeSize original(joined);
    // Selected loop entry costs have already been charged. Only version a
    // command in an ordinary gap, before the candidate is measured.
    auto replacement = CaseCodeSize(shared);
    const bool factored = selected_loop.empty() && replacement.commands == original.commands &&
                          replacement.guards <= original.guards &&
                          replacement.expressions <= original.expressions && FactorCommandGap(shared, work_);
    if (factored) replacement = CaseCodeSize(shared);
    // Charge inspections even on refusal. Crossing a selected-candidate
    // boundary additionally preserves its charged raw descriptor bound. Ordinary
    // gap sharing keeps its existing policy: arm sequence wrappers can grow
    // even when fewer statements will be materialized.
    work_.schedule_inspected_descriptors += original.statements + replacement.statements;
    work_.schedule_inspected_expressions += original.expressions + replacement.expressions;
    const bool within_budget = selected_loop.empty() || (replacement.expressions <= original.expressions &&
                                                         replacement.statements <= original.statements);
    if (factored || (replacement.commands == original.commands && replacement.guards < original.guards &&
                     within_budget)) {
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
    // A value-producing payload has an after/before gap just like an
    // effect-only call. Ordinary scalar definitions remain fences.
    const auto assignment = As<AssignStmt>(stmt);
    auto found = cuts_.find(assignment ? As<Call>(assignment->value_).get() : SingletonCall(stmt));
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

// Preserve expanded-tree growth bounds without visiting exponentially many
// paths through shared expression DAGs. Statement occurrences remain distinct;
// only context-independent expression subtree weights are memoized. Saturation
// is accompanied by an explicit unproved-size flag and cannot certify a fit.
class ScheduleTreeSize : public IRVisitor {
 public:
  size_t nodes = 0, payloads = 0, commands = 0, expressions = 0, statements = 0;
  size_t inspected_expressions = 0;
  bool overflow = false;
  void VisitExpr(const ExprPtr& expression) override {
    auto found = weights_.find(expression.get());
    if (found == weights_.end()) {
      ++inspected_expressions;
      const Weight previous{nodes, expressions, overflow};
      nodes = expressions = 1;
      overflow = false;
      IRVisitor::VisitExpr(expression);
      const Weight weight{nodes, expressions, overflow};
      found = weights_.emplace(expression.get(), weight).first;
      nodes = previous.nodes;
      expressions = previous.expressions;
      overflow = previous.overflow;
    }
    Add(nodes, found->second.nodes);
    Add(expressions, found->second.expressions);
    overflow |= found->second.overflow;
  }
  void VisitStmt(const StmtPtr& statement) override {
    Add(nodes, 1);
    Add(statements, 1);
    IRVisitor::VisitStmt(statement);
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (call && !IsOp(call, "system.set_flag") && !IsOp(call, "system.wait_flag") &&
        !core_affinity::IsPipeBarrier(call))
      Add(payloads, 1);
    else if (call)
      Add(commands, 1);
    IRVisitor::VisitStmt_(statement);
  }

 private:
  struct Weight {
    size_t nodes, expressions;
    bool overflow;
  };
  std::unordered_map<const Expr*, Weight> weights_;
  void Add(size_t& total, size_t value) {
    if (value > std::numeric_limits<size_t>::max() - total) {
      total = std::numeric_limits<size_t>::max();
      overflow = true;
    } else {
      total += value;
    }
  }
};

// Costs are accumulated by interval range additions over invocation cases.
// A mixed predicate can depend on an enclosing loop coordinate: its test
// executes, while all source work in its arms is possible rather than certain.
struct ScheduleCost {
  std::array<__int128, 6> value{};
  std::array<__int128, 6> uncertainty{};
  bool valid = true;
  bool Exact() const {
    return valid && std::all_of(uncertainty.begin(), uncertainty.end(), [](__int128 n) { return n == 0; });
  }
  __int128 Total() const { return std::accumulate(value.begin(), value.end(), __int128(0)); }
  __int128 UpperTotal() const { return std::accumulate(uncertainty.begin(), uncertainty.end(), Total()); }
  __int128 Tests(bool upper) const {
    const auto known = value[3] + value[4];
    return upper ? known + uncertainty[3] + uncertainty[4] : known;
  }
};
struct ScheduleCostProfile {
  std::vector<ScheduleCost> difference;
  bool valid = true;
  explicit ScheduleCostProfile(size_t cases) : difference(cases + 1) {}
  void Add(const ScheduleCost& cost, const Intervals& active, bool possible = false) {
    valid &= cost.valid;
    for (const auto& [lo, hi] : active)
      for (size_t i = 0; i < cost.value.size(); ++i) {
        auto& first = possible ? difference[lo].uncertainty : difference[lo].value;
        auto& end = possible ? difference[hi + 1].uncertainty : difference[hi + 1].value;
        first[i] += cost.value[i];
        end[i] -= cost.value[i];
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
    if (As<Ne>(expression)) {
      if (a->lo == a->hi && b->lo == b->hi)
        truth = a->lo != b->lo;
      else if (a->hi < b->lo || b->hi < a->lo)
        truth = true;
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
              const Intervals& active, bool possible = false) {
    if (active.empty()) {
      return;
    }
    for (const auto& command : forest.commands) {
      Visit(command, profile, weight, active, possible);
    }
  }
  std::optional<Intervals> ActiveArm(const FrozenCommand& command, const Intervals& active) {
    const auto& node = command.original;
    Intervals yes;
    const auto condition = Evaluate(command.condition_expr);
    const bool is_split_wrapper =
        command.body && command.body->commands.size() == 1 && command.body->commands[0].original == node;
    if (condition && condition->lo == condition->hi) {
      if (condition->lo) yes = active;
    } else {
      std::optional<InvocationCasePartition::Decision> direct_partition;
      if (partition_) {
        const auto before = work_.invocation_schedule_visits;
        direct_partition = partition_->Partition(command.condition_expr);
        work_.schedule_inspected_descriptors += work_.invocation_schedule_visits - before;
      }
      if (direct_partition) {
        yes = InvocationCasePartition::Intersection(active, direct_partition->yes);
      } else if (!is_split_wrapper && !node->guard.IsOpaque() && !node->guard.IsEmpty() &&
                 !node->guard.IsTautology() && node->guard.Arena() && partition_) {
        const auto& arena = node->guard.Arena();
        std::optional<AxisId> loop_ax;
        for (AxisId ax : ReferencedAxes(node->guard)) {
          const auto& info = arena->Axis(ax);
          if (info.kind == AxisKind::LoopFrame && info.loop == loop_.loop) {
            loop_ax = ax;
            break;
          }
        }
        RadixBoxGuard specialized = node->guard;
        if (loop_ax) {
          const auto phase_ap =
              AxisAP::MakeCanonical(first_, last_ + 1, stride_, arena->Axis(*loop_ax).count);
          if (!phase_ap) {
            return std::nullopt;
          }
          specialized = RestrictGuardAxis(node->guard, *loop_ax, *phase_ap, arena);
        }
        bool only_block = true;
        for (AxisId ax : ReferencedAxes(specialized)) {
          if (arena->Axis(ax).kind != AxisKind::LaunchBlock) {
            only_block = false;
            break;
          }
        }
        if (!only_block) {
          return std::nullopt;
        }
        if (specialized.IsTautology()) {
          yes = active;
        } else if (!specialized.IsEmpty()) {
          const ExprPtr block_expr = LowerRadixGuardToExpr(specialized, RadixBoxGuard::True(), node->span);
          const auto before = work_.invocation_schedule_visits;
          const auto partition = partition_->Partition(block_expr);
          work_.schedule_inspected_descriptors += work_.invocation_schedule_visits - before;
          if (!partition) {
            return std::nullopt;
          }
          yes = InvocationCasePartition::Intersection(active, partition->yes);
        }
      } else {
        return std::nullopt;
      }
    }
    return yes;
  }
  void Visit(const FrozenCommand& command, ScheduleCostProfile& profile, __int128 weight,
             const Intervals& active, bool possible) {
    ++work_.schedule_inspected_descriptors;
    if (!command.condition_expr) {
      if (command.body) {
        Forest(*command.body, profile, weight, active, possible);
      }
      return;
    }
    ScheduleCost local;
    for (size_t i = 0; i < command.scalar_cost.size(); ++i) {
      local.value[i] = weight * command.scalar_cost[i];
    }
    profile.Add(local, active, possible);
    if (const auto yes = ActiveArm(command, active)) {
      Forest(*command.body, profile, weight, *yes, possible);
      Forest(*command.alternative, profile, weight, SubtractIntervals(active, *yes), possible);
    } else {
      // Each arm can execute at most once for each test. Charge both to the
      // upper bound without enumerating outer-loop visits or combinations.
      Forest(*command.body, profile, weight, active, true);
      Forest(*command.alternative, profile, weight, active, true);
    }
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
           ScheduleCostProfile& profile, bool possible = false) {
    if (first > last) return;
    first_ = first;
    last_ = last;
    stride_ = stride;
    const __int128 weight = (last - first) / stride + 1;
    Forest(*FreezeCommandForest(forest, work_), profile, weight, active, possible);
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
    std::vector<const Call*> anchor_order;
    anchor_order.reserve(assigned.cuts.size());
    for (const auto& sc : assigned.cuts) anchor_order.push_back(sc.original.get());
    candidate.plan.groups =
        GroupInvocationSchedules(phases, assigned.block, domains, candidate.work, anchor_order);
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
      if (policy.boundary_motion) PrepareBoundaryMotion(loop, phase, assigned.arena, candidate.work);
      for (auto& cuts : phase.visits) CommandSeams(cuts, candidate.work).Run(loop.loop->body_);
      auto finalize = [&](ControlForest& forest) {
        if (!forest.empty()) forest = ShareCommandGuards(forest, candidate.work, false, true);
      };
      finalize(phase.entry);
      finalize(phase.exit);
      for (auto& cuts : phase.visits) {
        std::set<const PhysicalCuts*> seen;
        for (auto& [anchor, cut] : cuts) {
          if (!seen.insert(cut.get()).second) continue;
          finalize(cut->before);
          finalize(cut->after);
        }
      }
    }
    candidate.plan.versions.push_back(std::move(selected));
  }
  return candidate;
}

CandidateSchedule PrepareOrderPreservingBoundary(const CandidateSchedule& baseline,
                                                 const std::shared_ptr<GuardDomainArena>& arena) {
  CandidateSchedule candidate;
  const auto& loop = baseline.plan.loop;
  candidate.plan.loop = loop;
  candidate.plan.mode = kBoundaryMode;
  const auto& guarded = baseline.plan.versions.front().front();
  auto& phases = candidate.plan.versions.emplace_back();
  const auto axis = arena->RegisterLoopFrame(SyncLoopFrame{loop.loop, loop.count, loop.start, loop.step, {}});
  for (const auto& [first, end] : loop.phases) {
    PreparedSchedulePhase phase{first, end, 1, false, {}};
    auto& cuts = phase.visits.emplace_back(CopyScheduleCuts(guarded.visits.front()));
    const AxisAP range{first, end, 1};
    std::set<const PhysicalCuts*> seen;
    for (auto& [anchor, cut] : cuts) {
      if (!seen.insert(cut.get()).second) {
        continue;
      }
      // Restrict the final guarded forest in place order. Normalizing a
      // fresh forest after projection could commute independent commands
      // whose guards now make that profitable, changing their source order.
      cut->before = RestrictForestAxis(cut->before, axis, range, arena, candidate.work);
      cut->after = RestrictForestAxis(cut->after, axis, range, arena, candidate.work);
    }
    phases.push_back(std::move(phase));
  }
  phases.front().entry = guarded.entry;
  phases.back().exit = guarded.exit;
  return candidate;
}

// This conservative expanded bound includes every payload operand occurrence,
// not just descriptors or operation sites. Three times the command tree covers
// sequence wrappers and first-use SSA bindings retained by the frozen scalar plan.
bool BoundCandidate(CandidateSchedule& candidate, const ScheduleTreeSize& body,
                    const AssignedRegionSchedule& assigned, std::pair<int64_t, int64_t> blocks) {
  if (body.overflow) return false;
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
        nodes += __int128(body.nodes) + 8;  // Fresh offset binding and residue sequence.
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
    if (size.overflow) return false;
    nodes += __int128(size.nodes) + 4;
    candidate.work.schedule_inspected_expressions += size.inspected_expressions;
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
                      std::pair<int64_t, int64_t> blocks, ScheduleCostModel& model, bool bounded_model) {
  const auto& loop = candidate.plan.loop;
  ScheduleCostProfile profile(assigned.cases.size());
  std::set<const Call*> initialization_anchors;
  if (bounded_model) {
    for (const auto& indices : loop.case_cuts) {
      for (const auto index : indices) {
        const auto& cut = assigned.cuts[index];
        if (cut.continuation) {
          initialization_anchors.insert(cut.original.get());
          initialization_anchors.insert(cut.continuation.get());
        }
      }
    }
  }
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
      // A parent seam can share tests in a phase entry or exit with its
      // neighbors. Use these costs only in the conservative upper bound.
      model.Add(phase.entry, phase.first, phase.first, 1, active, profile, bounded_model);
      model.Add(phase.exit, phase.end - 1, phase.end - 1, 1, active, profile, bounded_model);
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
            // Emission can absorb a first-visit test into an original
            // accumulator initialization branch at this cut.
            model.Add(cut->before, first, last, stride, active, profile,
                      initialization_anchors.count(anchor) != 0);
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
      total.uncertainty[i] += profile.difference[c].uncertainty[i];
      if (total.value[i] + total.uncertainty[i] > std::numeric_limits<int64_t>::max() || total.value[i] < 0 ||
          total.uncertainty[i] < 0) {
        total.valid = false;
      } else {
        row[i] = static_cast<int64_t>(total.value[i]);
      }
    }
    candidate.receipt.model_cost.push_back(row);
    candidate.costs.push_back(total);
  }
  candidate.receipt.model_exact = std::all_of(candidate.costs.begin(), candidate.costs.end(),
                                              [](const auto& cost) { return cost.Exact(); });
}

bool Comparable(const CandidateSchedule& a, const CandidateSchedule& b) {
  if (a.costs.size() != b.costs.size()) return false;
  for (size_t i = 0; i < a.costs.size(); ++i) {
    if (!a.costs[i].Exact() || !b.costs[i].Exact()) {
      return false;
    }
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

bool ProvesBoundarySaving(const CandidateSchedule& candidate, const CandidateSchedule& baseline) {
  if (candidate.costs.empty() || candidate.costs.size() != baseline.costs.size()) {
    return false;
  }
  for (size_t c = 0; c < candidate.costs.size(); ++c) {
    const auto& before = baseline.costs[c];
    const auto& after = candidate.costs[c];
    // Charge loop headers as well as decision sites, and require a strict
    // saving even when every uncertain test in the candidate executes.
    if (!before.valid || !after.valid || after.Tests(true) >= before.Tests(false) ||
        after.UpperTotal() >= before.Total()) {
      return false;
    }
  }
  return true;
}

class ScheduleLoopOrder : public IRVisitor {
  size_t depth_ = 0;

 public:
  std::vector<std::pair<const ForStmt*, size_t>> loops;
  void VisitStmt_(const ForStmtPtr& loop) override {
    loops.emplace_back(loop.get(), depth_);
    ++depth_;
    IRVisitor::VisitStmt_(loop);
    --depth_;
  }
};
unsigned TransformCount(unsigned mode) { return (mode & 1) + ((mode >> 1) & 1) + ((mode >> 2) & 1); }
PreparedLoops PlanLoopSchedules(const AssignedRegionSchedule& assigned, const CutInventory& domains,
                                const RegularSchedules& inventory, std::pair<int64_t, int64_t> blocks,
                                PhysicalSyncRealization& work, const SyncSchedulePolicy& policy) {
  const auto requested = std::find(kScheduleModes.begin(), kScheduleModes.end(), policy.mode);
  const bool automatic = IsAutomaticSchedule(policy);
  const bool boundary_auto = policy.mode == kAutoBoundaryMode;
  CHECK(automatic || requested != kScheduleModes.end()) << "Unknown synchronization schedule mode";
  PreparedLoops selected;
  size_t payload_debit = 0, control_debit = 0;
  ScheduleLoopOrder order;
  order.VisitStmt(assigned.owner->body_);
  for (size_t ordinal = 0; ordinal < order.loops.size(); ++ordinal) {
    const auto [key, depth] = order.loops[ordinal];
    const auto found = inventory.find(key);
    if (found == inventory.end()) continue;
    const auto& loop = found->second;
    const auto label = loop.loop->loop_var_->name_hint_ + "#" + std::to_string(ordinal);
    ScheduleTreeSize original, body;
    original.VisitStmt(loop.loop);
    body.VisitStmt(loop.loop->body_);
    work.schedule_inspected_descriptors += original.statements + body.statements;
    work.schedule_inspected_expressions += original.inspected_expressions + body.inspected_expressions;
    const auto unproved_size = [&]() {
      SyncScheduleCandidateReceipt guarded{
          label, "guarded", "fallback",
          "expanded schedule size is unproved; preserve original guarded emission"};
      guarded.selected = true;
      guarded.payload_sites = original.payloads;
      guarded.control_node_bound = std::numeric_limits<size_t>::max();
      work.schedule_choices.push_back(std::move(guarded));
      for (unsigned mode = 1; mode < kScheduleModes.size(); ++mode) {
        if (!RequestsScheduleMode(policy, mode)) {
          continue;
        }
        work.schedule_choices.push_back(
            {label, kScheduleModes[mode], "cost_unproved",
             "expanded size cannot certify the complete-candidate growth budget"});
      }
    };
    if (original.overflow || body.overflow) {
      unproved_size();
      continue;
    }
    ScheduleCostModel model(loop, assigned, blocks, work);
    std::vector<CandidateSchedule> candidates;
    candidates.reserve(kScheduleModes.size());
    candidates.push_back(PrepareCandidate(loop, 0, assigned, domains, blocks, policy));
    auto& baseline = candidates.front();
    baseline.receipt = {label, kScheduleModes[0], "eligible", ""};
    if (!BoundCandidate(baseline, body, assigned, blocks)) {
      // The generic guarded emitter is already qualified by the immutable
      // assigned plan. No optional source duplication is selected without a
      // representable complete-candidate bound.
      work.schedule_inspected_descriptors +=
          baseline.work.schedule_inspected_descriptors + baseline.work.control_descriptors +
          baseline.work.schedule_descriptor_visits + baseline.work.schedule_body_visits +
          baseline.work.case_sequence_visits + baseline.work.case_sequence_comparisons +
          baseline.work.cut_case_comparisons + baseline.work.invocation_schedule_visits +
          baseline.work.invocation_schedule_comparisons;
      work.schedule_inspected_expressions +=
          baseline.work.schedule_inspected_expressions + baseline.work.control_expression_visits;
      unproved_size();
      continue;
    }
    MeasureCandidate(baseline, assigned, blocks, model, boundary_auto);
    baseline.receipt.model_comparable = Comparable(baseline, baseline);
    size_t chosen = 0;
    int64_t longest_boundary = 0;
    for (const auto& [first, end] : loop.phases) longest_boundary = std::max(longest_boundary, end - first);
    // Short loops can still remove repeated synchronization tests when
    // every phase is a single visit. Keep other short shapes guarded.
    const bool short_with_repeated_phase = loop.count < kMinAutoBoundaryTrips && longest_boundary > 1;
    for (unsigned mode = 1; mode < kScheduleModes.size(); ++mode) {
      if (!RequestsScheduleMode(policy, mode)) {
        continue;
      }
      CandidateSchedule candidate;
      candidate.receipt = {label, kScheduleModes[mode], "ineligible", ""};
      if ((mode & 1) && (assigned.cases.size() < 2 || !assigned.block))
        candidate.receipt.reason = "case versioning needs multiple cases and a dominating launch binding";
      else if ((mode & 2) && loop.phases.size() < 2)
        candidate.receipt.reason = "no internal endpoint boundary";
      else if (boundary_auto &&
               (depth == 0 || short_with_repeated_phase || loop.phases.size() > kMaxAutoBoundaryPhases))
        candidate.receipt.reason = "outside the bounded boundary-specialization shape";
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
      candidate = boundary_auto ? PrepareOrderPreservingBoundary(baseline, assigned.arena)
                                : PrepareCandidate(loop, mode, assigned, domains, blocks, policy);
      candidate.receipt = {label, kScheduleModes[mode], "eligible", ""};
      if ((mode & 1) && candidate.plan.groups.size() < 2) {
        candidate.receipt.status = "ineligible";
        candidate.receipt.reason = "fewer than two distinct complete-case schedules";
      } else if (!BoundCandidate(candidate, body, assigned, blocks) ||
                 candidate.receipt.payload_sites > payload_limit ||
                 __int128(candidate.receipt.control_node_bound) >
                     __int128(policy.max_extra_control_nodes) - control_debit ||
                 (boundary_auto && __int128(candidate.receipt.control_node_bound) >
                                       2 * __int128(baseline.receipt.control_node_bound))) {
        candidate.receipt.status = "over_budget";
        candidate.receipt.reason = "expanded emitted-site bound exceeds remaining budget";
      } else {
        MeasureCandidate(candidate, assigned, blocks, model, boundary_auto);
        candidate.receipt.model_comparable = Comparable(candidate, candidates.front());
        const bool profitable = boundary_auto ? ProvesBoundarySaving(candidate, candidates.front())
                                              : Improves(candidate, candidates.front());
        if (boundary_auto) {
          candidate.receipt.model_comparable |= profitable;
        }
        if (!automatic) {
          chosen = candidates.size();
        } else if (profitable &&
                   (chosen == 0 || Improves(candidate, candidates[chosen]) ||
                    (Comparable(candidate, candidates[chosen]) &&
                     std::equal(candidate.costs.begin(), candidate.costs.end(),
                                candidates[chosen].costs.begin(),
                                [](const auto& a, const auto& b) { return a.Total() == b.Total(); }) &&
                     std::make_tuple(candidate.receipt.payload_sites, candidate.receipt.control_node_bound,
                                     TransformCount(mode)) <
                         std::make_tuple(candidates[chosen].receipt.payload_sites,
                                         candidates[chosen].receipt.control_node_bound,
                                         TransformCount(candidates[chosen].plan.mode))))) {
          chosen = candidates.size();
        }
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
    accepted.receipt.status = automatic || chosen == 0 ? "selected" : "selected_forced";
    if (!automatic && accepted.receipt.mode != policy.mode) {
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
        saving = std::min(saving, candidates.front().costs[c].Total() - accepted.costs[c].UpperTotal());
      if (saving > 0 && saving <= std::numeric_limits<size_t>::max() - work.regular_schedule_model_saving)
        work.regular_schedule_model_saving += static_cast<size_t>(saving);
    }
    accepted.plan.receipt_label = label;
    selected.emplace(key, std::move(accepted.plan));
  }
  work.schedule_payload_growth_debit = payload_debit;
  work.schedule_control_growth_debit = control_debit;
  return selected;
}

// Boundary motion is also selected before materialization. Unsupported outer
// control retains the same conservative rule; no emitter discovers a rewrite.
class BoundaryScheduleCollector : public IRVisitor {
  const CutInventory& cuts_;
  const PreparedLoops& loops_;
  PhysicalSyncRealization& work_;
  std::shared_ptr<GuardDomainArena> arena_;
  void VisitStmt_(const ForStmtPtr& loop) override {
    if (loops_.count(loop.get())) return;
    auto plan = PlanBoundarySchedule(loop, cuts_, work_, arena_);
    if (!plan.entry.empty() || !plan.exit.empty()) plans.emplace(loop.get(), std::move(plan));
    IRVisitor::VisitStmt_(loop);
  }

 public:
  BoundarySchedules plans;
  BoundaryScheduleCollector(const CutInventory& cuts, const PreparedLoops& loops,
                            PhysicalSyncRealization& work, std::shared_ptr<GuardDomainArena> arena = nullptr)
      : cuts_(cuts), loops_(loops), work_(work), arena_(std::move(arena)) {}
};

void FinalizePostBoundaryGuards(CutInventory& cuts, BoundarySchedules& boundaries,
                                PhysicalSyncRealization& work) {
  auto finalize = [&](ControlForest& forest) {
    if (!forest.empty()) forest = ShareCommandGuards(forest, work, false, true);
  };
  std::set<const PhysicalCuts*> seen;
  for (auto& [anchor, cut] : cuts) {
    if (!seen.insert(cut.get()).second) continue;
    finalize(cut->before);
    finalize(cut->after);
  }
  for (auto& [loop, plan] : boundaries) {
    finalize(plan.entry);
    finalize(plan.exit);
    for (auto& [cut_ptr, pair] : plan.cuts) {
      finalize(pair.first);
      finalize(pair.second);
    }
  }
}

FunctionPtr CompileAssignedSyncSchedule(const FunctionPtr& owner, const std::vector<AssignedSyncCase>& cases,
                                        std::pair<int64_t, int64_t> blocks,
                                        const SyncSchedulePolicy& policy) {
  AssignedRegionSchedule schedule(owner, cases);
  // All invocation cases share one emitted body and one growth budget. Store
  // the complete expenditure on its first case so callers charge it once.
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
  CutDomains(schedule.owner, absent, domains, blocks, work, schedule.block).VisitStmt(schedule.owner->body_);
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
  BoundaryScheduleCollector boundary(cuts, plans, work, schedule.arena);
  if (policy.boundary_motion) boundary.VisitStmt(schedule.owner->body_);
  CommandSeams(cuts, work, &boundary.plans, &plans).Run(schedule.owner->body_);
  FinalizePostBoundaryGuards(cuts, boundary.plans, work);
  auto function = MaterializeSelectedSchedule(schedule.owner, schedule.block, cuts, plans, boundary.plans,
                                              blocks, work, names);
  FinishSyncScheduleReceipt(function, work, policy);
  return function;
}

void PopulateCutDomains(const FunctionPtr& owner, CutInventory& cuts, PhysicalSyncRealization& work) {
  const AbsentLoops absent;
  CutDomains(owner, absent, cuts, {0, 0}, work).VisitStmt(owner->body_);
}

BoundarySchedules PlanCommandBoundaries(const FunctionPtr& owner, CutInventory& cuts, bool enabled,
                                        PhysicalSyncRealization& work,
                                        const std::shared_ptr<GuardDomainArena>& arena) {
  PreparedLoops loops;
  BoundaryScheduleCollector boundary(cuts, loops, work, arena);
  if (enabled) {
    boundary.VisitStmt(owner->body_);
    CommandSeams(cuts, work, &boundary.plans, &loops).Run(owner->body_);
    FinalizePostBoundaryGuards(cuts, boundary.plans, work);
  }
  return std::move(boundary.plans);
}

}  // namespace physical

using physical::IsAutomaticSchedule;
using physical::kScheduleModes;
using physical::ScheduleTreeSize;
void FinishSyncScheduleReceipt(const FunctionPtr& function, PhysicalSyncRealization& work,
                               const SyncSchedulePolicy& policy) {
  const bool automatic = IsAutomaticSchedule(policy);
  CHECK(automatic ||
        std::find(kScheduleModes.begin(), kScheduleModes.end(), policy.mode) != kScheduleModes.end())
      << "Unknown synchronization schedule mode";
  if (work.schedule_choices.empty()) {
    if (!automatic && policy.mode != "guarded") {
      work.schedule_choices.push_back(
          {"<function>", policy.mode, "ineligible", "no qualified repeated loop"});
    }
    SyncScheduleCandidateReceipt identity{"<function>", "guarded",
                                          automatic || policy.mode == "guarded" ? "selected" : "fallback",
                                          "no qualified repeated loop"};
    identity.selected = true;
    work.schedule_choices.push_back(std::move(identity));
  }
  ScheduleTreeSize measured;
  measured.VisitFunction(function);
  work.schedule_inspected_descriptors += measured.statements;
  work.schedule_inspected_expressions += measured.inspected_expressions;
  work.schedule_emitted_payload_sites = measured.payloads;
  work.schedule_emitted_control_nodes = measured.nodes;
  for (auto& receipt : work.schedule_choices) {
    if (receipt.loop != "<function>" || !receipt.selected) continue;
    receipt.payload_sites = measured.payloads;
    receipt.control_node_bound = measured.nodes;
    receipt.command_sites = measured.commands;
    if (measured.overflow) {
      receipt.status = "cost_unproved";
      receipt.reason = "expanded emitted size exceeds the count representation; guarded plan retained";
      receipt.model_comparable = receipt.model_exact = false;
    }
  }
  if (measured.overflow &&
      std::none_of(work.schedule_choices.begin(), work.schedule_choices.end(),
                   [](const auto& receipt) { return receipt.loop == "<function>" && receipt.selected; })) {
    SyncScheduleCandidateReceipt unknown{"<function>", "guarded", "cost_unproved",
                                         "expanded emitted size exceeds the count representation"};
    unknown.selected = true;
    unknown.payload_sites = measured.payloads;
    unknown.control_node_bound = measured.nodes;
    unknown.command_sites = measured.commands;
    work.schedule_choices.push_back(std::move(unknown));
  }
}

}  // namespace pypto::ir::sync
