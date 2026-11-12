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

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_placement.h"

namespace pypto::ir::sync {
namespace {
using Direction = std::pair<PipeType, PipeType>;
using Family = PhysicalSyncFamily;

Occurrence Endpoint(const Family& f, int64_t ordinal, bool publish) {
  auto visit = publish ? f.source_first : f.target_first;
  if (f.axes.empty()) {
    visit += ordinal;
  } else {
    for (auto axis = f.axes.rbegin(); axis != f.axes.rend(); ++axis) {
      visit += (ordinal % axis->count) * (publish ? axis->source_stride : axis->target_stride);
      ordinal /= axis->count;
    }
    INTERNAL_CHECK(!ordinal) << "Internal error: scoped physical record exceeds domain";
  }
  return {publish ? f.source_region : f.target_region, visit, publish ? f.source_site : f.target_site};
}
Occurrence Source(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, true); }
Occurrence Target(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, false); }
Direction Pipes(const CompactSyncInput& a, const Family& f) {
  return {a.segments[f.source_region].sites[f.source_site].pipe,
          a.segments[f.target_region].sites[f.target_site].pipe};
}

// This is a narrow binding of the existing native resource declarations, not
// an inference that arbitrary missing catalog entries mean signal-free code.
// Direct Buffer IR emits TLOAD(Mat/Vec), TSTORE(Acc/Vec), TMOV(Mat->Left/Right),
// TEXTRACT(Mat->Left/Right), TMATMUL and TMATMUL_ACC for these admitted calls.
// In manual a2a3 mode those variants have no hidden AIC flags. TEXTRACT's
// catalogued V<->S claims belong to the excluded Vec overload. Still consume
// any applicable fixed claims for the owning core conservatively, and refuse
// unresolved owners. Vec arithmetic takes its macro from the shared recipe.
std::map<Direction, std::set<int>> NativeClaims(const CompactSyncInput& a, const std::string& target) {
  CHECK(target == "a2a3") << "Unsupported physical target: native variants are qualified only for a2a3";
  auto backend = backend::GetBackendInstance(backend::BackendType::Ascend910);
  std::map<Direction, std::set<int>> claimed;
  std::set<std::string> checked;
  for (const auto& segment : a.segments) {
    if (!segment.count) continue;
    for (const auto& site : segment.sites) {
      for (const auto& call : {site.original, site.continuation}) {
        if (!call) continue;
        std::string macro;
        if (IsOp(call, "buffer.load")) {
          macro = "pto::TLOAD";
        } else if (IsOp(call, "buffer.store")) {
          macro = "pto::TSTORE";
        } else if (IsOp(call, "buffer.copy")) {
          macro = "pto::TMOV";
        } else if (IsOp(call, "buffer.extract")) {
          macro = "pto::TEXTRACT";
        } else if (IsOp(call, "buffer.matmul")) {
          macro = "pto::TMATMUL";
        } else if (IsOp(call, "buffer.matmul_acc")) {
          macro = "pto::TMATMUL_ACC";
        } else {
          // The same recipe selects the emitted PTO operation. Resource claims
          // remain authoritative; unresolved ISA-helper ownership still refuses.
          const char* native = nullptr;
          if (auto recipe = backend::FindBufferElementwiseRecipe(call->op_->name_)) {
            native = recipe->native_op;
          } else if (auto recipe = backend::FindBufferIndexedRecipe(call->op_->name_)) {
            native = recipe->native_op;
          } else if (auto recipe = backend::FindBufferRowRecipe(call->op_->name_)) {
            native = recipe->native_op;
          }
          CHECK(native) << "Unavailable native synchronization resource binding: " << call->op_->name_;
          macro = std::string(native).substr(4);
          std::transform(macro.begin(), macro.end(), macro.begin(),
                         [](unsigned char c) { return std::toupper(c); });
          macro = "pto::" + macro;
        }
        if (!checked.insert(macro).second) continue;
        const auto declaration = backend->FindIsaMacroResources(macro);
        if (!declaration) continue;  // Only the enumerated, audited native variants.
        for (const auto& claim : declaration->claims) {
          using pypto::sync::CoreScope;
          using pypto::sync::IsaIdSource;
          using pypto::sync::IsaResourceLifetime;
          using pypto::sync::IsaSyncMode;
          using pypto::sync::ResourceKind;
          if (std::find(claim.modes.begin(), claim.modes.end(), IsaSyncMode::Manual) == claim.modes.end())
            continue;
          if (macro == "pto::TEXTRACT" && claim.pool.core == CoreScope::AIV) continue;
          CHECK(claim.pool.kind == ResourceKind::PipelineEvent &&
                claim.pool.core ==
                    (a.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC) &&
                claim.source == IsaIdSource::Fixed && claim.lifetime == IsaResourceLifetime::Invocation)
              << "Unavailable native flag owner/variant binding: " << macro;
          for (auto id : claim.ids) claimed[{claim.pool.src, claim.pool.dst}].insert(static_cast<int>(id));
        }
      }
    }
  }
  return claimed;
}

class Allocator {
 public:
  Allocator(const CompactSyncInput& input, std::vector<Family> families, const DirectSyncOrder& order,
            const std::string& target, const std::optional<std::vector<int>>& restricted)
      : input_(input),
        order_(order),
        catalog_(pypto::sync::FindResourceCatalog(target)),
        restricted_(restricted),
        claimed_(SyncNativeClaims(input, target)) {
    result_.families = std::move(families);
    result_.finite_minimum = order_.finite_minimum && !input_.region_tree;
    CHECK(order_.precedes && order_.reference_before && order_.threshold)
        << "Missing direct synchronization order proof";
    for (const auto& f : result_.families) {
      CHECK(f.source_region < input_.segments.size() && f.target_region < input_.segments.size())
          << "Invalid direct family region";
      const auto& source = input_.segments[f.source_region];
      const auto& target_segment = input_.segments[f.target_region];
      CHECK(f.source_site < source.sites.size() && f.target_site < target_segment.sites.size() &&
            f.source_first >= 0 && f.target_first >= 0 && f.count > 0 && f.ids.empty())
          << "Invalid direct family endpoint or preassigned IDs";
      __int128 last_source = f.source_first, last_target = f.target_first, volume = 1;
      if (f.axes.empty()) {
        last_source += f.count - 1;
        last_target += f.count - 1;
      } else {
        CHECK(f.axes.size() <= source.nested_frames.size() &&
              f.axes.size() <= target_segment.nested_frames.size())
            << "Invalid direct family arity";
        for (size_t i = 0; i < f.axes.size(); ++i) {
          const auto& axis = f.axes[i];
          CHECK(axis.frame == source.nested_frames[i].loop &&
                axis.frame == target_segment.nested_frames[i].loop && axis.count > 0 &&
                axis.count <= source.nested_frames[i].count &&
                axis.count <= target_segment.nested_frames[i].count && axis.source_stride >= 0 &&
                axis.target_stride >= 0)
              << "Invalid direct family scope";
          volume *= axis.count;
          CHECK(volume <= std::numeric_limits<int64_t>::max()) << "Direct family volume overflow";
          last_source += __int128(axis.count - 1) * axis.source_stride;
          last_target += __int128(axis.count - 1) * axis.target_stride;
          CHECK(last_source < source.count && last_target < target_segment.count)
              << "Direct family axis exceeds occurrence domain";
        }
        CHECK(volume == f.count) << "Inconsistent direct family volume";
      }
      CHECK(last_source < source.count && last_target < target_segment.count)
          << "Direct family endpoint exceeds occurrence domain";
    }
    CHECK(catalog_) << "Unavailable synchronization resource catalog: " << target;
    CHECK(input_.hardware_target == target)
        << "Synchronization hardware contract differs from lowering target";
    if (restricted_) {
      std::set<int> unique;
      for (int id : *restricted_)
        CHECK(id >= 0 && unique.insert(id).second) << "Invalid/duplicate eligible ID";
    }
  }
  PhysicalSyncRealization Run() {
    for (const auto& [direction, ids] : claimed_)
      for (int id : ids) result_.native_reserved_ids.emplace_back(direction.first, direction.second, id);
    QualifyBarriers();
    for (size_t index = 0; index < result_.families.size(); ++index) {
      auto& f = result_.families[index];
      const auto direction = Pipes(input_, f);
      if (direction.first == direction.second) continue;
      const auto& ids = Eligible(direction);
      int64_t width = f.count;
      if (f.count > 1 && input_.region_tree) {
        result_.finite_minimum = false;
        CHECK(order_.scoped_reuse && order_.scoped_reuse(index))
            << "Scoped period-one physical reuse is unproved; no capacity obstruction established";
        width = 1;
      } else if (f.count > 1) {
        result_.finite_minimum = false;
        INTERNAL_CHECK(f.source_region == f.target_region)
            << "Internal error: repeated family without a periodic owner";
        const auto distance = order_.threshold(f.source_region, f.target_site, f.source_site);
        // Canonical direct placement associates WAIT(h)->SET(g) with
        // C_consumer(h)->I_source(g). Reference-forward witnesses between valid
        // endpoints stay inside the finite prefix. Clip before adding a wide
        // threshold, so huge reachable distances never overflow or mean absence.
        const int64_t displacement = f.target_first - f.source_first;
        INTERNAL_CHECK(displacement >= 0) << "Negative direct-family displacement";
        if (distance && *distance < f.count && displacement < f.count) {
          const __int128 period = __int128(displacement) + *distance;
          width = static_cast<int64_t>(std::min<__int128>(f.count, std::max<__int128>(1, period)));
        }
      }
      CHECK(width <= static_cast<int64_t>(ids.size()))
          << (f.count == 1 ? "Fixed-plan scarcity: direction has no eligible ID"
                           : "Allocation unproved: family lane bound exceeds eligible capacity")
          << " direction=" << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
          << " lanes=" << width << " eligible=" << ids.size();
      f.ids.resize(width, -1);
      for (int64_t residue = 0; residue < width; ++residue) {
        const int64_t last = residue + ((f.count - 1 - residue) / width) * width;
        lanes_[direction].push_back(
            {index, static_cast<size_t>(residue), Source(f, residue), Target(f, last)});
        ++result_.reserved_lanes;
      }
    }
    for (auto& [direction, lanes] : lanes_) Color(direction, lanes);
    return std::move(result_);
  }

 private:
  struct Lane {
    size_t family, residue;
    Occurrence first, last;
  };
  const CompactSyncInput& input_;
  const DirectSyncOrder& order_;
  const pypto::sync::ResourceCatalog* catalog_;
  const std::optional<std::vector<int>>& restricted_;
  std::map<Direction, std::set<int>> claimed_;
  std::map<Direction, std::vector<int>> eligible_;
  std::map<Direction, std::vector<Lane>> lanes_;
  PhysicalSyncRealization result_;
  bool Reaches(const Occurrence& a, const Occurrence& b) {
    ++result_.reuse_queries;
    return order_.precedes(a, b);
  }
  const std::vector<int>& Eligible(Direction d) {
    auto [it, fresh] = eligible_.try_emplace(d);
    if (!fresh) return it->second;
    it->second = SyncEligibleIds(input_, input_.hardware_target, d, restricted_, claimed_);
    return it->second;
  }
  void QualifyBarriers() {
    if (input_.region_tree) {
      for (size_t i = 0; i < result_.families.size(); ++i) {
        const auto [source, target] = Pipes(input_, result_.families[i]);
        if (source != target) continue;
        CHECK(target == PipeType::M || target == PipeType::V)
            << "Scoped same-pipe barrier mechanism is unavailable; scalar barriers are forbidden";
        CHECK(order_.scoped_barrier && order_.scoped_barrier(i))
            << "Scoped same-pipe barrier adjacency is unproved";
      }
      return;
    }
    // Predecessor lookup is indexed by region and pipe. For repeated bodies,
    // check both startup and steady-state; no trip count is unfolded.
    std::vector<std::array<std::optional<Occurrence>, kPipeCount>> before(input_.segments.size());
    std::array<std::optional<Occurrence>, kPipeCount> previous{};
    std::vector<std::vector<std::optional<size_t>>> local(input_.segments.size());
    std::vector<std::array<std::optional<size_t>, kPipeCount>> last(input_.segments.size());
    for (size_t r = 0; r < input_.segments.size(); ++r) {
      before[r] = previous;
      local[r].resize(input_.segments[r].sites.size());
      if (!input_.segments[r].count) continue;
      for (size_t s = 0; s < input_.segments[r].sites.size(); ++s) {
        auto pipe = input_.segments[r].sites[s].pipe;
        local[r][s] = last[r][pipe];
        last[r][pipe] = s;
        previous[pipe] = Occurrence{r, input_.segments[r].count - 1, s};
      }
    }
    for (const auto& f : result_.families) {
      auto [src, dst] = Pipes(input_, f);
      if (src != dst) continue;
      CHECK(dst != PipeType::S) << "Scalar barriers are forbidden; require qualified native scalar order";
      CHECK(dst == PipeType::M || dst == PipeType::V) << "Unavailable same-pipe barrier mechanism";
      for (int64_t offset : std::set<int64_t>{0, f.count - 1}) {
        auto [r, v, s] = Target(f, offset);
        std::optional<Occurrence> pred;
        if (local[r][s])
          pred = Occurrence{r, v, *local[r][s]};
        else if (v)
          pred = Occurrence{r, v - 1, *last[r][dst]};
        else
          pred = before[r][dst];
        CHECK(pred && Reaches(*pred, Target(f, offset)))
            << "Unavailable order-equal same-pipe barrier placement";
      }
    }
  }
  void Color(Direction direction, std::vector<Lane>& lanes) {
    // Sort two boundary chains once, then sweep the monotone first-reuse
    // suffix. Queries are charged separately (the retained regional oracle
    // need not be constant-time). No quadratic directory scan or K unfolding.
    auto by_source = [&](const Lane& x, const Lane& y) {
      if (x.first != y.first) return order_.reference_before(x.first, y.first);
      return std::tie(x.family, x.residue) < std::tie(y.family, y.residue);
    };
    std::sort(lanes.begin(), lanes.end(), by_source);
    const auto colors = ColorOrderedSyncLanes(
        lanes.size(), direction,
        [&](size_t x, size_t y) {
          if (lanes[x].last != lanes[y].last) return order_.reference_before(lanes[x].last, lanes[y].last);
          return std::tie(lanes[x].family, lanes[x].residue) < std::tie(lanes[y].family, lanes[y].residue);
        },
        [&](size_t x, size_t y) { return Reaches(lanes[x].last, lanes[y].first); }, Eligible(direction),
        result_.finite_minimum);
    for (size_t i = 0; i < lanes.size(); ++i) {
      const auto& lane = lanes[i];
      result_.families[lane.family].ids[lane.residue] = colors[i];
    }
  }
};

struct PhysicalCuts {
  std::vector<StmtPtr> before, after;
};
using CutInventory = std::map<const Call*, std::shared_ptr<PhysicalCuts>>;

class Emitter : public SyncPlacement {
 public:
  Emitter(const CompactSyncInput& input, PhysicalSyncRealization& result)
      : SyncPlacement(input), result_(result) {
    for (size_t i = 0; i < result.families.size(); ++i) {
      const auto& f = result.families[i];
      incoming_[{f.target_region, f.target_site}].push_back(i);
      if (!f.ids.empty()) outgoing_[{f.source_region, f.source_site}].push_back(i);
    }
  }

  void AppendCuts(CutInventory& inventory, const ExprPtr& guard) {
    for (size_t r = 0; r < input_.segments.size(); ++r) {
      const auto& region = input_.segments[r];
      if (!region.count) continue;
      for (size_t s = 0; s < region.sites.size(); ++s) {
        const auto& site = region.sites[s];
        auto commands = Commands({r, s}, site.original->span_);
        if (commands.before.empty() && commands.after.empty()) continue;
        auto& retained = inventory[site.original.get()];
        if (!retained) retained = std::make_shared<PhysicalCuts>();
        if (site.continuation) {
          auto& continuation = inventory[site.continuation.get()];
          INTERNAL_CHECK(!continuation || continuation == retained)
              << "Internal error: incompatible original initialization cut";
          continuation = retained;
        }
        auto append = [&](std::vector<StmtPtr>& target, const std::vector<StmtPtr>& source) {
          if (source.empty()) return;
          if (guard)
            target.push_back(Guard(guard, std::make_shared<SeqStmts>(source, site.original->span_)));
          else
            target.insert(target.end(), source.begin(), source.end());
        };
        append(retained->before, commands.before);
        append(retained->after, commands.after);
      }
    }
  }

 private:
  PhysicalSyncRealization& result_;
  std::map<SyncSiteKey, std::vector<size_t>> incoming_, outgoing_;
  ExprPtr Integer(int64_t n, const Span& span) {
    return std::make_shared<ConstInt>(n, DataType::INDEX, span);
  }
  ExprPtr Ordinal(size_t region, const Span& span) {
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
  std::vector<ExprPtr> ScopedGuards(const Family& family, bool publish, const Span& span) {
    const auto region = publish ? family.source_region : family.target_region;
    auto base = publish ? family.source_first : family.target_first;
    const auto& frames = input_.segments[region].nested_frames;
    std::vector<int64_t> digits(frames.size());
    for (size_t i = frames.size(); i-- > 0;) {
      INTERNAL_CHECK(frames[i].count > 0) << "Internal error: physical endpoint in absent frame";
      digits[i] = base % frames[i].count;
      base /= frames[i].count;
    }
    INTERNAL_CHECK(!base && family.axes.size() <= frames.size()) << "Internal error: physical scope arity";
    std::vector<ExprPtr> guards;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto count = i < family.axes.size() ? family.axes[i].count : 1;
      if (i < family.axes.size())
        INTERNAL_CHECK(family.axes[i].frame == frames[i].loop) << "Internal error: physical scope identity";
      if (!digits[i] && count == frames[i].count) continue;
      const auto value = FrameDigit(frames[i], span);
      if (count == 1) {
        guards.push_back(std::make_shared<Eq>(value, Integer(digits[i], span), DataType::BOOL, span));
      } else {
        if (digits[i])
          guards.push_back(std::make_shared<Ge>(value, Integer(digits[i], span), DataType::BOOL, span));
        if (digits[i] + count < frames[i].count)
          guards.push_back(
              std::make_shared<Lt>(value, Integer(digits[i] + count, span), DataType::BOOL, span));
      }
    }
    return guards;
  }
  StmtPtr Guard(ExprPtr condition, StmtPtr body) {
    ++result_.guard_sites;
    return std::make_shared<IfStmt>(condition, body, std::nullopt, std::vector<VarPtr>{}, body->span_);
  }
  StmtPtr Command(const Family& f, bool publish, const Span& span) {
    const auto [src, dst] = Pipes(input_, f);
    const size_t r = publish ? f.source_region : f.target_region;
    const int64_t first = publish ? f.source_first : f.target_first;
    const bool scoped = input_.region_tree.has_value();
    // The initial scoped certificate deliberately proves period one. Keep
    // physical table coordinates separate from the full logical record tuple.
    INTERNAL_CHECK(!scoped || f.ids.size() <= 1) << "Internal error: unsupported scoped physical period";
    const auto ordinal = scoped ? Integer(0, span) : Ordinal(r, span);
    ExprPtr index = ordinal;
    if (!scoped && first) index = std::make_shared<Sub>(index, Integer(first, span), DataType::INDEX, span);
    std::vector<StmtPtr> commands;
    if (src == dst) {
      INTERNAL_CHECK(dst == PipeType::M || dst == PipeType::V)
          << "Only qualified M/V barriers may be emitted; scalar barriers are forbidden";
      auto call = OpRegistry::GetInstance().Create(dst == PipeType::M ? "system.bar_m" : "system.bar_v", {},
                                                   {}, span);
      commands.push_back(std::make_shared<EvalStmt>(call, span));
      ++result_.barrier_sites;
    } else {
      ExprPtr residue;
      if (f.ids.size() > 1)
        residue = std::make_shared<FloorMod>(index, Integer(f.ids.size(), span), DataType::INDEX, span);
      for (size_t i = 0; i < f.ids.size(); ++i) {
        auto call = OpRegistry::GetInstance().Create(
            publish ? "system.set_flag" : "system.wait_flag", {},
            {{"src_pipe", static_cast<int>(src)},
             {"dst_pipe", static_cast<int>(dst)},
             {"event_id", f.ids[i]},
             {"core_type", std::string(input_.owner->func_type_ == FunctionType::AIV ? "aiv" : "aic")}},
            span);
        StmtPtr command = std::make_shared<EvalStmt>(call, span);
        if (residue)
          command = Guard(std::make_shared<Eq>(residue, Integer(i, span), DataType::BOOL, span), command);
        commands.push_back(command);
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      }
    }
    StmtPtr body = std::make_shared<SeqStmts>(commands, span);
    // The range guard surrounds residue arithmetic: no absent predecessor is
    // evaluated/consumed. Final-only exports never publish on earlier visits.
    if (scoped) {
      auto guards = ScopedGuards(f, publish, span);
      for (auto guard = guards.rbegin(); guard != guards.rend(); ++guard) body = Guard(*guard, body);
      return body;
    }
    if (first + f.count < input_.segments[r].count)
      body = Guard(std::make_shared<Lt>(ordinal, Integer(first + f.count, span), DataType::BOOL, span), body);
    if (first) body = Guard(std::make_shared<Ge>(ordinal, Integer(first, span), DataType::BOOL, span), body);
    return body;
  }
  PhysicalCuts Commands(SyncSiteKey site, const Span& span) {
    PhysicalCuts cuts;
    // Preserve the common direct protocol: local barrier, incoming WAITs,
    // payload, outgoing SETs. Case aggregation never interleaves active cases.
    for (auto i : incoming_[site])
      if (result_.families[i].ids.empty()) cuts.before.push_back(Command(result_.families[i], false, span));
    for (auto i : incoming_[site])
      if (!result_.families[i].ids.empty()) cuts.before.push_back(Command(result_.families[i], false, span));
    for (auto i : outgoing_[site]) cuts.after.push_back(Command(result_.families[i], true, span));
    return cuts;
  }
  StmtPtr Wrap(const StmtPtr& payload, SyncSiteKey site) override {
    auto cuts = Commands(site, payload->span_);
    cuts.before.push_back(payload);
    cuts.before.insert(cuts.before.end(), cuts.after.begin(), cuts.after.end());
    return std::make_shared<SeqStmts>(cuts.before, payload->span_);
  }
};
// The command builder above owns all protocol mechanics. This adapter only
// installs already-qualified case cuts in the unchanged original payload tree.
class InvocationEmitter : public IRMutator {
  const CutInventory& inventory_;
  StmtPtr Wrap(const StmtPtr& payload, const PhysicalCuts& cuts) {
    auto statements = cuts.before;
    statements.push_back(payload);
    statements.insert(statements.end(), cuts.after.begin(), cuts.after.end());
    return std::make_shared<SeqStmts>(statements, payload->span_);
  }
  static const Call* ArmCall(StmtPtr arm) {
    if (auto seq = As<SeqStmts>(arm)) {
      if (seq->stmts_.size() != 1) return nullptr;
      arm = seq->stmts_.front();
    }
    const auto eval = As<EvalStmt>(arm);
    return eval ? As<Call>(eval->expr_).get() : nullptr;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    const auto found = inventory_.find(call.get());
    return found == inventory_.end() ? statement : Wrap(statement, *found->second);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& statement) override {
    const auto found = inventory_.find(ArmCall(statement->then_body_));
    if (found == inventory_.end()) return IRMutator::VisitStmt_(statement);
    CHECK(statement->else_body_) << "Invocation cut requires the original initialization alternatives";
    const auto other = inventory_.find(ArmCall(*statement->else_body_));
    CHECK(other != inventory_.end() && other->second == found->second)
        << "Invocation initialization alternatives have incompatible command cuts";
    return Wrap(statement, *found->second);
  }

 public:
  explicit InvocationEmitter(const CutInventory& inventory) : inventory_(inventory) {}
};

std::vector<Family> DirectFamilies(const CompactSyncResult& analysis) {
  PhysicalSyncRealization result_;
  if (analysis.input.region_tree) {
    result_.finite_minimum = false;  // Sufficient whole-span assignment, not a minimum-ID proof.
    const auto owned = analysis.ExportScopedDemands();
    CHECK(owned) << "Scoped endpoint export is unavailable";
    for (const auto& schema : owned->families) {
      const auto [sr, sv, ss] = schema.source;
      const auto [tr, tv, ts] = schema.target;
      __int128 count = 1;
      for (const auto& axis : schema.axes) {
        count *= axis.count;
        CHECK(count <= std::numeric_limits<int64_t>::max()) << "Scoped physical record domain overflow";
      }
      result_.families.push_back({sr, ss, tr, ts, sv, tv, static_cast<int64_t>(count), {}, schema.axes});
    }
  } else {
    const auto crossings = analysis.ExportCrossingDemands();
    CHECK(crossings.has_value())
        << "Minimum demands are retained; shared-repetition physical endpoint scope is unavailable";
    for (size_t r = 0; r < analysis.segments.size(); ++r) {
      const auto& segment = analysis.segments[r];
      for (const auto& f : segment.demands) {
        CHECK(f.distance >= 0 && f.source < segment.pipes.size() && f.target < segment.pipes.size())
            << "Invalid physical demand family";
        if (segment.count <= f.distance) continue;
        result_.families.push_back({r, f.source, r, f.target, 0, f.distance, segment.count - f.distance, {}});
      }
    }
    for (const auto& [source, target] : *crossings) {
      auto [sr, sv, ss] = source;
      auto [tr, tv, ts] = target;
      CHECK(sr < analysis.segments.size() && tr < analysis.segments.size() && sr < tr)
          << "Unavailable physical crossing frame";
      CHECK(sv >= 0 && sv < analysis.segments[sr].count && tv >= 0 && tv < analysis.segments[tr].count &&
            ss < analysis.segments[sr].pipes.size() && ts < analysis.segments[tr].pipes.size())
          << "Invalid physical crossing occurrence";
      result_.families.push_back({sr, ss, tr, ts, sv, tv, 1, {}});
    }
  }
  return std::move(result_.families);
}
DirectSyncOrder MathematicalOrder(const CompactSyncResult& analysis) {
  return {[&](const Occurrence& a, const Occurrence& b) { return analysis.Precedes(a, true, b, false); },
          [&](const Occurrence& a, const Occurrence& b) { return analysis.ReferenceBefore(a, b); },
          [&](size_t r, size_t a, size_t b) { return analysis.segments[r].Threshold(a, true, b, false); },
          [&](size_t i) { return analysis.ScopedConsecutiveReuse(i); },
          [&](size_t i) { return analysis.ScopedAdjacentBarrier(i); },
          true};
}

}  // namespace
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target) {
  return NativeClaims(input, target);
}
std::vector<int> SyncEligibleIds(const CompactSyncInput& input, const std::string& target,
                                 SyncDirection direction, const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed) {
  using pypto::sync::CoreScope;
  using pypto::sync::ResourceKind;
  const auto* catalog = pypto::sync::FindResourceCatalog(target);
  CHECK(catalog) << "Unavailable synchronization resource catalog: " << target;
  CHECK(input.hardware_target == target) << "Synchronization hardware contract differs from lowering target";
  const auto core = input.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC;
  auto spec = catalog->FindPool({ResourceKind::PipelineEvent, core, direction.first, direction.second});
  CHECK(spec) << "Unavailable directional event pool";
  auto legal = spec->legal_ids;
  if (legal.empty()) {
    legal.resize(spec->capacity);
    std::iota(legal.begin(), legal.end(), 0);
  }
  std::set<int> available;
  const auto owners = claimed.find(direction);
  for (auto id : legal)
    if (std::find(spec->reserved_ids.begin(), spec->reserved_ids.end(), id) == spec->reserved_ids.end() &&
        (owners == claimed.end() || !owners->second.count(static_cast<int>(id))))
      available.insert(static_cast<int>(id));
  std::vector<int> result;
  if (restricted) {
    std::set<int> unique;
    for (int id : *restricted) {
      CHECK(id >= 0 && unique.insert(id).second) << "Invalid/duplicate eligible ID";
      CHECK(id < static_cast<int>(spec->capacity) && std::find(legal.begin(), legal.end(), id) != legal.end())
          << "Requested ID is outside the legal target pool";
      if (available.count(id)) result.push_back(id);
    }
  } else {
    result.assign(available.begin(), available.end());
  }
  return result;
}
std::vector<int> ColorOrderedSyncLanes(size_t count, SyncDirection direction,
                                       const std::function<bool(size_t, size_t)>& wait_before,
                                       const std::function<bool(size_t, size_t)>& reusable,
                                       const std::vector<int>& eligible, bool finite_minimum) {
  std::vector<size_t> waits(count), release(count);
  std::iota(waits.begin(), waits.end(), 0);
  std::sort(waits.begin(), waits.end(), wait_before);
  size_t next = 0;
  for (auto i : waits) {
    while (next < count && !reusable(i, next)) ++next;
    release[i] = next;
    INTERNAL_CHECK(next > i) << "Internal error: cyclic physical lane span";
  }
  std::vector<std::vector<int>> buckets(count + 1);
  std::vector<int> free, result(count);
  size_t colors = 0;
  for (size_t i = 0; i < count; ++i) {
    free.insert(free.end(), buckets[i].begin(), buckets[i].end());
    if (free.empty()) {
      CHECK(colors < eligible.size())
          << (finite_minimum ? "Fixed-plan scarcity" : "Allocation unproved by whole-span reservation")
          << ": direction=" << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
          << " overlap_at_publication=" << i << " required_at_least=" << colors + 1
          << " eligible=" << eligible.size();
      free.push_back(eligible[colors++]);
    }
    const int id = free.back();
    free.pop_back();
    result[i] = id;
    buckets[release[i]].push_back(id);
  }
  return result;
}
PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted) {
  return Allocator(input, std::move(families), order, target, restricted).Run();
}
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted) {
  auto result = AllocateDirectSync(input, std::move(families), order, target, restricted);
  result.function = Emitter(input, result).VisitFunction(input.owner);
  return result;
}
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids) {
  return RealizeDirectSync(analysis.input, DirectFamilies(analysis), MathematicalOrder(analysis), target,
                           eligible_ids);
}
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases) {
  CHECK(owner && !cases.empty()) << "Missing complete invocation case owner";
  const auto block = cases.size() > 1 ? DominatingSyncBlock(owner) : nullptr;
  CutInventory cuts;
  auto next = blocks.first;
  for (const auto& item : cases) {
    CHECK(item.input && item.plan && item.input->owner == owner &&
          item.input->parameter_bounds == parameters && item.input->block_bounds &&
          item.input->block_bounds->first == next && item.input->block_bounds->second >= next &&
          item.input->block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Invocation cases must retain exhaustive original owners and contracts";
    next = item.input->block_bounds->second + 1;
    const auto guard = SyncInvocationGuard(block, blocks, *item.input->block_bounds, owner->span_);
    Emitter(*item.input, *item.plan).AppendCuts(cuts, guard);
  }
  CHECK(next == blocks.second + 1) << "Incomplete invocation activation domain";
  // Exclusive cases share palettes only within this original invocation.
  // Guarded cuts contain endpoint/ID arithmetic as well as SET/WAIT commands.
  return InvocationEmitter(cuts).VisitFunction(owner);
}
PhysicalSyncInvocation RealizePhysicalInvocation(const SyncInvocationAnalysis& analysis,
                                                 const std::string& target,
                                                 const std::optional<std::vector<int>>& eligible_ids) {
  // Qualify executable case selection before spending work on allocation.
  CHECK(analysis.cases.size() <= 1 || DominatingSyncBlock(analysis.owner))
      << "Original dominating invocation block binding is unavailable";
  PhysicalSyncInvocation result{analysis, nullptr, {}};
  result.cases.reserve(analysis.cases.size());
  std::vector<AssignedSyncCase> cases;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics &&
          item.mathematics->input.block_bounds == std::make_pair(item.block_first, item.block_last))
        << "Missing original invocation analysis";
    result.cases.push_back(AllocateDirectSync(item.mathematics->input, DirectFamilies(*item.mathematics),
                                              MathematicalOrder(*item.mathematics), target, eligible_ids));
    cases.push_back({&item.mathematics->input, &result.cases.back()});
  }
  result.function =
      EmitAssignedSyncCases(analysis.owner, analysis.parameter_bounds, analysis.block_bounds, cases);
  return result;
}

FunctionPtr SyncOutcome(const FunctionPtr& function, bool pypto, const std::string& reason) {
  auto attrs = function->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& attr) {
                               return attr.first == kAttrPyptoAutosync ||
                                      attr.first == kAttrAutosyncFallbackReason;
                             }),
              attrs.end());
  attrs.emplace_back(kAttrPyptoAutosync, pypto);
  if (!pypto) attrs.emplace_back(kAttrAutosyncFallbackReason, reason);
  return std::make_shared<Function>(function->name_, function->params_, function->param_directions_,
                                    function->return_types_, function->body_, function->span_,
                                    function->func_type_, function->level_, function->role_, attrs,
                                    function->requires_runtime_binding_, function->ir_stage_);
}
}  // namespace pypto::ir::sync

namespace pypto::ir::pass {
Pass RealizeSync() {
  auto transform = [](const FunctionPtr& function, const sync::SyncParameterBounds& bounds,
                      const std::optional<std::pair<int64_t, int64_t>>& block_bounds) -> FunctionPtr {
    if (!IsInCoreType(function->func_type_)) return function;
    // These functions do not enter PTO code generation; their external owner
    // remains responsible for synchronization. Do not claim PTOAS delegation.
    if (function->requires_runtime_binding_ || function->HasAttr(kAttrExternalSource) ||
        function->HasAttr(kAttrBuiltinTemplateDir))
      return function;
    auto* ctx = PassContext::Current();
    CHECK(ctx && ctx->GetEnablePyptoAutoSync()) << "RealizeSync requires enable_pypto_autosync=True";
    CHECK(!function->GetAttr<bool>(kAttrPyptoAutosync))
        << "RealizeSync cannot reprocess an owned physical plan";
    if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV)
      return sync::SyncOutcome(function, false, "PyPTO synchronization supports only AIC/AIV leaves");
    if (function->ir_stage_ != FunctionIRStage::Buffer || ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS)
      return sync::SyncOutcome(function, false, "PyPTO synchronization requires physically placed Buffer IR");
    if (!ctx->GetAutoSyncDisjointGM())
      return sync::SyncOutcome(function, false, "Disjoint GM argument contract was not supplied");
    auto target = ctx->GetBackendHandler()->GetPtoTargetArch();
    if (target != "a2a3")
      return sync::SyncOutcome(function, false,
                               "PyPTO physical synchronization supports only manual a2a3 macros");
    // Only publish after complete analysis AND realization. A qualification or
    // resource refusal returns the original body, never partially emitted IR.
    // Internal errors are deliberately not caught. The strict retained-result
    // APIs still throw on these user-facing input/proof refusals.
    try {
      FunctionPtr emitted;
      size_t set_sites = 0, wait_sites = 0;
      if (block_bounds) {
        // The complete invocation owner qualifies all activation cases before
        // any command cuts are published. Do not pick one sampled worker.
        auto analysis = sync::AnalyzeSyncInvocation(function, true, target, bounds, *block_bounds);
        auto physical = sync::RealizePhysicalInvocation(analysis, target, std::nullopt);
        emitted = physical.function;
        for (const auto& item : physical.cases) {
          set_sites += item.set_sites;
          wait_sites += item.wait_sites;
        }
      } else {
        auto analysis = sync::AnalyzeCompactSync(sync::ImportPlacedSync(function, true, target, bounds));
        auto physical = sync::RealizePhysicalSync(analysis, target, std::nullopt);
        emitted = physical.function;
        set_sites = physical.set_sites;
        wait_sites = physical.wait_sites;
      }
      // PTOAS recognizes literal flags, not empty plans or barriers, as manual
      // ownership. Do not fabricate a dummy flag or mix partial ownership.
      if (!set_sites || !wait_sites)
        return sync::SyncOutcome(function, false,
                                 "Candidate has no SET/WAIT marker; delegate the original kernel");
      return sync::SyncOutcome(emitted, true, "");
    } catch (const pypto::ValueError& error) {
      return sync::SyncOutcome(function, false, error.what());
    }
  };
  // Range evidence belongs to this closed program's call sites, not mutable
  // function metadata. Do not infer bounds when the standalone API has none.
  return CreateProgramPass(
      [transform](const ProgramPtr& program) {
        auto evidence = sync::CollectSyncInvocationEvidence(program);
        auto functions = program->functions_;
        for (auto& [global, function] : functions) {
          auto block = evidence.block_indices.find(global->name_);
          function = transform(
              function, evidence.parameters[global->name_],
              block == evidence.block_indices.end() ? std::nullopt : std::make_optional(block->second));
        }
        return std::make_shared<Program>(functions, program->name_, program->span_);
      },
      "RealizeSync", PassProperties{});
}
}  // namespace pypto::ir::pass
