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

#include "src/ir/transforms/utils/physical_sync_allocation.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/sync_placement.h"

namespace pypto::ir::sync {
namespace {
using Direction = SyncDirection;
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
    families_ = std::move(families);
    result_.finite_minimum = order_.finite_minimum && !input_.region_tree;
    CHECK(order_.precedes && order_.reference_before && order_.threshold)
        << "Missing direct synchronization order proof";
    for (const auto& f : families_) {
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
    for (size_t index = 0; index < families_.size(); ++index) {
      auto& f = families_[index];
      const auto direction = SyncFamilyPipes(input_, f);
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

  std::vector<Family> families_;

 public:
  std::vector<Family> TakeFamilies() { return std::move(families_); }
  std::map<Direction, std::vector<int>> TakeEligible() { return std::move(eligible_); }

 private:
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
      for (size_t i = 0; i < families_.size(); ++i) {
        const auto [source, target] = SyncFamilyPipes(input_, families_[i]);
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
    for (const auto& f : families_) {
      auto [src, dst] = SyncFamilyPipes(input_, f);
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
      families_[lane.family].ids[lane.residue] = colors[i];
    }
  }
};

}  // namespace
SyncDirection SyncFamilyPipes(const CompactSyncInput& a, const Family& f) {
  return {a.segments[f.source_region].sites[f.source_site].pipe,
          a.segments[f.target_region].sites[f.target_site].pipe};
}
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
namespace {
using FrameKey = std::tuple<ForStmtPtr, int64_t, int64_t, int64_t, ExprPtr>;
FrameKey AssignmentFrame(const SyncLoopFrame& frame) {
  return {frame.loop, frame.count, frame.start, frame.step, frame.start_expression};
}
using SiteKey = std::tuple<CallPtr, CallPtr, PipeType, ForStmtPtr>;
using SegmentKey =
    std::tuple<ForStmtPtr, int64_t, int64_t, int64_t, int64_t, std::vector<FrameKey>, std::vector<SiteKey>>;
std::vector<SegmentKey> AssignmentSegments(const CompactSyncInput& input) {
  std::vector<SegmentKey> result;
  for (const auto& segment : input.segments) {
    std::vector<FrameKey> frames;
    for (const auto& frame : segment.nested_frames) frames.push_back(AssignmentFrame(frame));
    std::vector<SiteKey> sites;
    for (const auto& site : segment.sites)
      sites.emplace_back(site.original, site.continuation, site.pipe, site.initialization_scope);
    result.emplace_back(segment.original_loop, segment.count, segment.start, segment.step,
                        segment.reset_period, std::move(frames), std::move(sites));
  }
  return result;
}
using RegionKey = std::tuple<std::optional<size_t>, size_t, std::optional<FrameKey>,
                             std::optional<std::pair<size_t, size_t>>>;
std::vector<RegionKey> AssignmentRegions(const CompactSyncInput& input) {
  std::vector<RegionKey> result;
  auto visit = [&](auto&& self, const SyncRegion& region) -> void {
    result.emplace_back(
        region.leaf, region.children.size(),
        region.repetition ? std::make_optional(AssignmentFrame(*region.repetition)) : std::nullopt,
        region.startup);
    for (const auto& child : region.children) self(self, child);
  };
  if (input.region_tree) visit(visit, *input.region_tree);
  return result;
}
}  // namespace

struct AssignedSyncPlan::State {
  const CompactSyncInput* input;
  FunctionPtr owner;
  std::string target;
  SyncParameterBounds parameters;
  std::optional<std::pair<int64_t, int64_t>> blocks;
  std::vector<SegmentKey> segments;
  std::vector<RegionKey> regions;
  std::vector<PhysicalSyncFamily> families;
  std::map<SyncDirection, std::vector<int>> eligible;
  std::vector<std::tuple<PipeType, PipeType, int>> reservations;
};
AssignedSyncPlan::AssignedSyncPlan(const CompactSyncInput& input, std::vector<PhysicalSyncFamily> families,
                                   std::map<SyncDirection, std::vector<int>> eligible,
                                   std::vector<std::tuple<PipeType, PipeType, int>> reservations)
    : state_(std::make_shared<State>(State{&input, input.owner, input.hardware_target, input.parameter_bounds,
                                           input.block_bounds, AssignmentSegments(input),
                                           AssignmentRegions(input), std::move(families), std::move(eligible),
                                           std::move(reservations)})) {
  // Allocation established matching/lifetime proofs. Freezing checks the assigned
  // resource inventory, not a second coloring or reconstructed order analysis.
  for (const auto& family : state_->families) {
    const auto direction = SyncFamilyPipes(input, family);
    if (direction.first == direction.second) {
      INTERNAL_CHECK(family.ids.empty()) << "Same-pipe family has physical flag IDs";
      continue;
    }
    const auto found = state_->eligible.find(direction);
    INTERNAL_CHECK(found != state_->eligible.end() && !family.ids.empty()) << "Missing assigned palette";
    for (int id : family.ids) {
      INTERNAL_CHECK(std::find(found->second.begin(), found->second.end(), id) != found->second.end())
          << "Assigned flag is outside the qualified directional pool";
      INTERNAL_CHECK(std::find(state_->reservations.begin(), state_->reservations.end(),
                               std::make_tuple(direction.first, direction.second, id)) ==
                     state_->reservations.end())
          << "Assigned flag collides with a native owner";
    }
  }
}
const std::vector<PhysicalSyncFamily>& AssignedSyncPlan::Families() const { return state_->families; }
void AssignedSyncPlan::ValidateFor(const CompactSyncInput& input, const std::string& target) const {
  CHECK(state_->input == &input) << "Assigned synchronization plan belongs to a different input case";
  CHECK(state_->owner == input.owner) << "Assigned synchronization plan has a foreign original owner";
  CHECK(state_->target == target && input.hardware_target == target)
      << "Assigned synchronization target differs from its certificate";
  CHECK(state_->parameters == input.parameter_bounds && state_->blocks == input.block_bounds)
      << "Assigned synchronization invocation domain changed";
  CHECK(state_->segments == AssignmentSegments(input) && state_->regions == AssignmentRegions(input))
      << "Assigned synchronization occurrence frame changed";
}
const std::vector<PhysicalSyncFamily>& PhysicalSyncRealization::Families() const {
  static const std::vector<PhysicalSyncFamily> empty;
  return assignment ? assignment->Families() : empty;  // Unassigned command diagnostics only.
}

PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted) {
  Allocator allocator(input, std::move(families), order, target, restricted);
  auto result = allocator.Run();
  result.assignment = std::shared_ptr<const AssignedSyncPlan>(new AssignedSyncPlan(
      input, allocator.TakeFamilies(), allocator.TakeEligible(), result.native_reserved_ids));
  return result;
}

}  // namespace pypto::ir::sync
