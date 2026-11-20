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
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/scoped_sync_palette.h"
#include "src/ir/transforms/utils/scoped_token_families.h"
#include "src/ir/transforms/utils/sync_footprint_resolver.h"
#include "src/ir/transforms/utils/sync_placement.h"

namespace pypto::ir::sync {
namespace {
using Direction = SyncDirection;
using Family = PhysicalSyncFamily;
Occurrence Endpoint(const Family& family, int64_t ordinal, SyncEndpointKind endpoint) {
  const auto occurrence =
      DecodeSyncFamilyEndpoint(family, ordinal, endpoint, SyncEmptyAxes::ContiguousVisits);
  CHECK(occurrence) << "Direct family record has incompatible coordinates";
  return *occurrence;
}
Occurrence Source(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, SyncEndpointKind::Source); }
Occurrence Target(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, SyncEndpointKind::Target); }

// Resource binding uses the same selected implementation as access import.
// Each distinct immutable call is resolved once, including continuation arms;
// do not deduplicate by macro alone because overload/core claims can differ.
void ReserveNativeClaims(const backend::BufferSyncImplementation& implementation,
                         SyncNativeReservations& claimed) {
  for (const auto& claim : implementation.Claims())
    for (auto id : claim.ids) claimed[{claim.pool.src, claim.pool.dst}].insert(static_cast<int>(id));
}
std::map<Direction, std::set<int>> NativeClaims(const CompactSyncInput& a, const std::string& target) {
  CHECK(a.hardware_target == target) << "Synchronization hardware contract differs from lowering target";
  backend::RequireLocalSyncExecutionContract(target);
  std::map<Direction, std::set<int>> claimed;
  std::set<const Call*> checked;
  for (const auto& segment : a.segments) {
    if (!segment.count) continue;
    for (const auto& site : segment.sites) {
      for (const auto& call : {site.original, site.continuation}) {
        if (!call || !checked.insert(call.get()).second) continue;
        const auto implementation =
            backend::SelectBufferSyncImplementation(call, a.owner->func_type_, target);
        INTERNAL_CHECK(implementation.Pipe() == site.pipe) << "Native binding differs from imported pipe";
        ReserveNativeClaims(implementation, claimed);
      }
    }
  }
  return claimed;
}

struct LaneColorAttempt {
  bool accepted;
  std::vector<int> colors;
  size_t at = 0, required = 0;
};
LaneColorAttempt TryColorOrderedSyncLanes(size_t count,
                                          const std::function<bool(size_t, size_t)>& wait_before,
                                          const std::function<bool(size_t, size_t)>& reusable,
                                          const std::vector<int>& eligible);
std::string ColorRefusal(SyncDirection direction, const LaneColorAttempt& attempt, size_t eligible,
                         bool finite_minimum) {
  std::ostringstream out;
  out << (finite_minimum ? "Fixed-plan scarcity" : "Allocation unproved by whole-span reservation")
      << ": direction=" << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
      << " overlap_at_publication=" << attempt.at << " required_at_least=" << attempt.required
      << " eligible=" << eligible;
  return out.str();
}

class Allocator {
 public:
  Allocator(const CompactSyncInput& input, std::vector<Family> families, const DirectSyncOrder& order,
            const std::string& target, const std::optional<std::vector<int>>& restricted,
            const SyncNativeReservations& extra_reservations = {})
      : input_(input),
        order_(order),
        catalog_(pypto::sync::FindResourceCatalog(target)),
        restricted_(restricted),
        claimed_(order.imported_native_claims ? *order.imported_native_claims
                                              : SyncNativeClaims(input, target)) {
    for (const auto& [direction, ids] : extra_reservations)
      claimed_[direction].insert(ids.begin(), ids.end());
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
    if (std::any_of(families_.begin(), families_.end(), [&](const auto& f) {
          const auto d = SyncFamilyPipes(input_, f);
          return f.count > 1 && d.first != d.second;
        }))
      result_.finite_minimum = false;
    std::map<Direction, std::vector<size_t>> directions;
    for (size_t i = 0; i < families_.size(); ++i) {
      const auto direction = SyncFamilyPipes(input_, families_[i]);
      if (direction.first != direction.second) directions[direction].push_back(i);
    }
    for (const auto& [direction, indices] : directions) {
      const auto before_queries = result_.reuse_queries;
      auto legacy = TryDirection(direction, indices);
      result_.allocation_attempts.push_back({direction.first, direction.second,
                                             legacy.status == LegacyStatus::Accepted        ? "accepted"
                                             : legacy.status == LegacyStatus::NoEligible    ? "no-eligible"
                                             : legacy.status == LegacyStatus::ExactScarcity ? "exact-scarcity"
                                                                                            : "unproved",
                                             legacy.reason, result_.reuse_queries - before_queries});
      if (legacy.status == LegacyStatus::Accepted) {
        for (const auto& [index, width] : legacy.widths) families_[index].ids.resize(width, -1);
        for (size_t i = 0; i < legacy.lanes.size(); ++i)
          families_[legacy.lanes[i].family].ids[legacy.lanes[i].residue] = legacy.colors[i];
        result_.reserved_lanes += legacy.lanes.size();
        result_.finite_minimum &= legacy.minimum;
        continue;
      }
      std::string palette_refusal;
      if (legacy.status == LegacyStatus::Unproved && order_.protocol && order_.scoped_palette) {
        const auto& ids = Eligible(direction);
        auto attempt = std::make_shared<const ScopedPaletteResult>(
            order_.scoped_palette(direction.first, direction.second, ids.size()));
        result_.palette_attempts.push_back(attempt);
        auto certificate = attempt->certificate;
        if (!certificate) {
          palette_refusal = "; palette status=" + std::string(ScopedPaletteStatusName(attempt->status)) +
                            ": " + attempt->reason;
        }
        if (certificate) {
          INTERNAL_CHECK(certificate->Schedule()->Protocol() == order_.protocol &&
                         certificate->Schedule()->SourcePipe() == direction.first &&
                         certificate->Schedule()->TargetPipe() == direction.second &&
                         certificate->Budget() > 0 &&
                         certificate->Budget() <= static_cast<int64_t>(ids.size()))
              << "Invalid complete-direction palette certificate";
          const auto width = static_cast<size_t>(certificate->Budget());
          palettes_.push_back({std::move(certificate), std::vector<int>(ids.begin(), ids.begin() + width)});
          result_.reserved_lanes += width;
          result_.finite_minimum = false;
          continue;
        }
      }
      CHECK(false) << legacy.reason << palette_refusal;
    }
    return std::move(result_);
  }

 private:
  struct Lane {
    size_t family, residue;
    Occurrence first, last;
  };
  enum class LegacyStatus { Accepted, Unproved, NoEligible, ExactScarcity };
  struct LegacyChoice {
    LegacyStatus status = LegacyStatus::Accepted;
    std::string reason;
    bool minimum = false;
    std::vector<Lane> lanes;
    std::vector<std::pair<size_t, int64_t>> widths;
    std::vector<int> colors;
  };
  const CompactSyncInput& input_;
  const DirectSyncOrder& order_;
  const pypto::sync::ResourceCatalog* catalog_;
  const std::optional<std::vector<int>>& restricted_;
  std::map<Direction, std::set<int>> claimed_;
  std::map<Direction, std::vector<int>> eligible_;
  PhysicalSyncRealization result_;

  std::vector<Family> families_;
  std::vector<AssignedScopedPalette> palettes_;

 public:
  std::vector<AssignedScopedPalette> TakePalettes() { return std::move(palettes_); }
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
      CHECK((order_.allow_stronger_barriers || dst == PipeType::M || dst == PipeType::V) &&
            RequireLocalSyncBarrier(dst, input_.owner->func_type_))
          << "Unavailable same-pipe barrier mechanism";
      for (int64_t offset : std::set<int64_t>{0, f.count - 1}) {
        auto [r, v, s] = Target(f, offset);
        std::optional<Occurrence> pred;
        if (local[r][s])
          pred = Occurrence{r, v, *local[r][s]};
        else if (v)
          pred = Occurrence{r, v - 1, *last[r][dst]};
        else
          pred = before[r][dst];
        CHECK(pred &&
              (order_.allow_stronger_barriers ? order_.reference_before(Source(f, offset), Target(f, offset))
                                              : Reaches(*pred, Target(f, offset))))
            << "Unavailable order-equal same-pipe barrier placement";
      }
    }
  }
  LegacyChoice TryDirection(Direction direction, const std::vector<size_t>& indices) {
    LegacyChoice choice;
    choice.minimum = result_.finite_minimum;
    const auto& ids = Eligible(direction);
    if (ids.empty()) {
      choice.status = LegacyStatus::NoEligible;
      choice.reason =
          "Fixed-plan scarcity: direction has no eligible ID direction=" + PipeTypeToString(direction.first) +
          "->" + PipeTypeToString(direction.second);
      return choice;
    }
    for (auto index : indices) {
      const auto& f = families_[index];
      int64_t width = f.count;
      if (f.count > 1 && input_.region_tree) {
        choice.minimum = false;
        if (!order_.scoped_reuse || !order_.scoped_reuse(index)) {
          choice.status = LegacyStatus::Unproved;
          choice.reason = "Scoped period-one physical reuse is unproved; no capacity obstruction established";
          return choice;
        }
        width = 1;
      } else if (f.count > 1) {
        choice.minimum = false;
        INTERNAL_CHECK(f.source_region == f.target_region)
            << "Internal error: repeated family without a periodic owner";
        const auto distance = order_.threshold(f.source_region, f.target_site, f.source_site);
        const int64_t displacement = f.target_first - f.source_first;
        INTERNAL_CHECK(displacement >= 0) << "Negative direct-family displacement";
        if (distance && *distance < f.count && displacement < f.count) {
          const __int128 period = __int128(displacement) + *distance;
          width = static_cast<int64_t>(std::min<__int128>(f.count, std::max<__int128>(1, period)));
        }
      }
      if (width > static_cast<int64_t>(ids.size())) {
        choice.status = LegacyStatus::Unproved;
        std::ostringstream out;
        out << "Allocation unproved: family lane bound exceeds eligible capacity direction="
            << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
            << " lanes=" << width << " eligible=" << ids.size();
        choice.reason = out.str();
        return choice;
      }
      choice.widths.emplace_back(index, width);
      for (int64_t residue = 0; residue < width; ++residue) {
        const int64_t last = residue + ((f.count - 1 - residue) / width) * width;
        choice.lanes.push_back({index, static_cast<size_t>(residue), Source(f, residue), Target(f, last)});
      }
    }
    auto& lanes = choice.lanes;
    std::sort(lanes.begin(), lanes.end(), [&](const Lane& x, const Lane& y) {
      if (x.first != y.first) return order_.reference_before(x.first, y.first);
      return std::tie(x.family, x.residue) < std::tie(y.family, y.residue);
    });
    auto colored = TryColorOrderedSyncLanes(
        lanes.size(),
        [&](size_t x, size_t y) {
          if (lanes[x].last != lanes[y].last) return order_.reference_before(lanes[x].last, lanes[y].last);
          return std::tie(lanes[x].family, lanes[x].residue) < std::tie(lanes[y].family, lanes[y].residue);
        },
        [&](size_t x, size_t y) { return Reaches(lanes[x].last, lanes[y].first); }, ids);
    if (!colored.accepted) {
      choice.status = choice.minimum ? LegacyStatus::ExactScarcity : LegacyStatus::Unproved;
      choice.reason = ColorRefusal(direction, colored, ids.size(), choice.minimum);
      return choice;
    }
    choice.colors = std::move(colored.colors);
    return choice;
  }
};

}  // namespace
SyncDirection SyncFamilyPipes(const CompactSyncInput& a, const Family& f) {
  CHECK(f.source_region < a.segments.size() && f.target_region < a.segments.size())
      << "Synchronization family region is unavailable";
  CHECK(f.source_site < a.segments[f.source_region].sites.size() &&
        f.target_site < a.segments[f.target_region].sites.size())
      << "Synchronization family site is unavailable";
  return {a.segments[f.source_region].sites[f.source_site].pipe,
          a.segments[f.target_region].sites[f.target_site].pipe};
}
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target) {
  return NativeClaims(input, target);
}
SyncNativeReservations SyncNativeClaims(const FunctionPtr& owner, const std::vector<CallPtr>& calls,
                                        const std::string& target) {
  backend::RequireLocalSyncExecutionContract(target);
  SyncNativeReservations claimed;
  SyncFootprintResolver resolver(owner, nullptr, /*run_scalar_layout_check=*/true, target);
  std::set<const Call*> checked;
  for (const auto& call : calls) {
    if (!checked.insert(call.get()).second) continue;
    const auto operation = backend::QualifyBufferSyncOperation(call, owner->func_type_, target);
    resolver.RequireVecCopyPlacement(operation);
    resolver.RequireWorkspaceSeparation(operation);
    ReserveNativeClaims(operation.Implementation(), claimed);
  }
  return claimed;
}
std::vector<int> SyncEligibleIds(const CompactSyncInput& input, const std::string& target,
                                 SyncDirection direction, const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed) {
  CHECK(input.hardware_target == target) << "Synchronization hardware contract differs from lowering target";
  return SyncEligibleIds(input.owner, target, direction, restricted, claimed);
}
std::vector<int> SyncEligibleIds(const FunctionPtr& owner, const std::string& target, SyncDirection direction,
                                 const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed) {
  using pypto::sync::CoreScope;
  using pypto::sync::ResourceKind;
  const auto* catalog = pypto::sync::FindResourceCatalog(target);
  CHECK(catalog) << "Unavailable synchronization resource catalog: " << target;
  CHECK(owner && (owner->func_type_ == FunctionType::AIV || owner->func_type_ == FunctionType::AIC))
      << "Directional resources require a qualified AIC/AIV owner";
  const auto core = owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC;
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
namespace {
LaneColorAttempt TryColorOrderedSyncLanes(size_t count,
                                          const std::function<bool(size_t, size_t)>& wait_before,
                                          const std::function<bool(size_t, size_t)>& reusable,
                                          const std::vector<int>& eligible) {
  std::vector<size_t> waits(count), release(count);
  std::iota(waits.begin(), waits.end(), 0);
  std::sort(waits.begin(), waits.end(), wait_before);
  size_t next = 0;
  for (auto i : waits) {
    while (next < count && !reusable(i, next)) ++next;
    release[i] = next;
    INTERNAL_CHECK(next > i) << "Internal error: cyclic physical lane span";
  }
  std::vector<std::vector<size_t>> buckets(count + 1);
  std::priority_queue<size_t, std::vector<size_t>, std::greater<size_t>> free_idx;
  std::vector<int> result(count);
  size_t colors = 0;
  for (size_t i = 0; i < count; ++i) {
    for (size_t idx : buckets[i]) free_idx.push(idx);
    if (free_idx.empty()) {
      if (colors >= eligible.size()) return {false, {}, i, colors + 1};
      free_idx.push(colors++);
    }
    const size_t idx = free_idx.top();
    free_idx.pop();
    result[i] = eligible[idx];
    buckets[release[i]].push_back(idx);
  }
  return {true, std::move(result), 0, 0};
}
}  // namespace
std::vector<int> ColorOrderedSyncLanes(size_t count, SyncDirection direction,
                                       const std::function<bool(size_t, size_t)>& wait_before,
                                       const std::function<bool(size_t, size_t)>& reusable,
                                       const std::vector<int>& eligible, bool finite_minimum) {
  auto result = TryColorOrderedSyncLanes(count, wait_before, reusable, eligible);
  CHECK(result.accepted) << ColorRefusal(direction, result, eligible.size(), finite_minimum);
  return std::move(result.colors);
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
  std::shared_ptr<const ScopedSyncProtocol> protocol;
  std::vector<size_t> protocol_assignments;
  std::vector<AssignedScopedPalette> palettes;
  std::vector<AssignedRawPalette> raw_palettes;
};
AssignedSyncPlan::AssignedSyncPlan(const CompactSyncInput& input, std::vector<PhysicalSyncFamily> families,
                                   std::map<SyncDirection, std::vector<int>> eligible,
                                   std::vector<std::tuple<PipeType, PipeType, int>> reservations,
                                   std::shared_ptr<const ScopedSyncProtocol> protocol,
                                   std::vector<AssignedScopedPalette> palettes) {
  auto state = std::make_shared<State>(State{&input,
                                             input.owner,
                                             input.hardware_target,
                                             input.parameter_bounds,
                                             input.block_bounds,
                                             AssignmentSegments(input),
                                             AssignmentRegions(input),
                                             std::move(families),
                                             std::move(eligible),
                                             std::move(reservations),
                                             std::move(protocol),
                                             {},
                                             std::move(palettes),
                                             {}});
  if (state->protocol) {
    state->protocol->ValidateFor(input);
    state->protocol_assignments.assign(state->protocol->Families().size(), state->families.size());
    for (size_t f = 0; f < state->families.size(); ++f) {
      const auto& assigned = state->families[f];
      CHECK(!assigned.protocol_uses.empty()) << "Assigned family lost its raw token uses";
      for (auto raw : assigned.protocol_uses) {
        CHECK(raw < state->protocol_assignments.size() &&
              state->protocol_assignments[raw] == state->families.size())
            << "Assigned protocol duplicates or invents a token use";
        const auto& use = state->protocol->Families()[raw];
        CHECK(use.source_region == assigned.source_region && use.source_site == assigned.source_site &&
              use.target_region == assigned.target_region && use.target_site == assigned.target_site)
            << "Assigned protocol changes an original command cut";
        state->protocol_assignments[raw] = f;
      }
    }
    CHECK(std::none_of(state->protocol_assignments.begin(), state->protocol_assignments.end(), [&](size_t f) {
      return f == state->families.size();
    })) << "Assigned protocol omits a token use";
  }
  if (!state->palettes.empty()) {
    INTERNAL_CHECK(state->protocol) << "Palette has no raw protocol";
    state->raw_palettes.resize(state->protocol->Families().size());
    std::set<SyncDirection> owned;
    for (size_t p = 0; p < state->palettes.size(); ++p) {
      const auto& palette = state->palettes[p];
      INTERNAL_CHECK(palette.certificate && palette.certificate->Schedule()->Protocol() == state->protocol)
          << "Foreign palette certificate";
      const auto& schedule = *palette.certificate->Schedule();
      const SyncDirection direction{schedule.SourcePipe(), schedule.TargetPipe()};
      INTERNAL_CHECK(owned.insert(direction).second &&
                     palette.ids.size() == static_cast<size_t>(palette.certificate->Budget()))
          << "Duplicate or malformed directional palette";
      std::set<int> unique;
      const auto& eligible = state->eligible.at(direction);
      for (int id : palette.ids)
        INTERNAL_CHECK(
            unique.insert(id).second && std::find(eligible.begin(), eligible.end(), id) != eligible.end() &&
            std::find(state->reservations.begin(), state->reservations.end(),
                      std::make_tuple(direction.first, direction.second, id)) == state->reservations.end())
            << "Palette ID collision or ineligible resource";
      for (size_t i = 0; i < schedule.Pieces().size(); ++i) {
        const auto raw = schedule.Pieces()[i].raw_use;
        INTERNAL_CHECK(raw < state->raw_palettes.size() &&
                       SyncFamilyPipes(input, state->protocol->Families()[raw]) == direction)
            << "Palette rank piece has foreign raw use";
        auto& mapping = state->raw_palettes[raw];
        INTERNAL_CHECK(!mapping.palette || *mapping.palette == p) << "Raw use has multiple physical owners";
        mapping.palette = p;
        mapping.pieces.push_back(i);
      }
      for (size_t raw = 0; raw < state->raw_palettes.size(); ++raw)
        if (SyncFamilyPipes(input, state->protocol->Families()[raw]) == direction) {
          INTERNAL_CHECK(state->raw_palettes[raw].palette == p && !state->raw_palettes[raw].pieces.empty())
              << "Directional palette omits raw token use";
          INTERNAL_CHECK(state->families[state->protocol_assignments[raw]].ids.empty())
              << "Palette direction also has a legacy physical owner";
        }
    }
  }
  state_ = std::move(state);
  // Allocation established matching/lifetime proofs. Freezing checks the assigned
  // resource inventory, not a second coloring or reconstructed order analysis.
  for (const auto& family : state_->families) {
    const auto direction = SyncFamilyPipes(input, family);
    if (direction.first == direction.second) {
      INTERNAL_CHECK(family.ids.empty()) << "Same-pipe family has physical flag IDs";
      continue;
    }
    if (!family.protocol_uses.empty() && !state_->raw_palettes.empty() &&
        state_->raw_palettes[family.protocol_uses.front()].palette)
      continue;
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
const std::shared_ptr<const ScopedSyncProtocol>& AssignedSyncPlan::Protocol() const {
  return state_->protocol;
}
const std::vector<size_t>& AssignedSyncPlan::ProtocolAssignments() const {
  return state_->protocol_assignments;
}
const std::vector<AssignedScopedPalette>& AssignedSyncPlan::Palettes() const { return state_->palettes; }
const std::vector<AssignedRawPalette>& AssignedSyncPlan::RawPalettes() const { return state_->raw_palettes; }
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

PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted,
                                           const SyncNativeReservations& extra_reservations) {
  if (order.protocol) {
    CHECK(order.validate_families) << "Missing retained protocol inventory validation";
    order.protocol->ValidateFor(input);
    order.validate_families(families);
  }
  Allocator allocator(input, std::move(families), order, target, restricted, extra_reservations);
  auto result = allocator.Run();
  result.assignment = std::shared_ptr<const AssignedSyncPlan>(
      new AssignedSyncPlan(input, allocator.TakeFamilies(), allocator.TakeEligible(),
                           result.native_reserved_ids, order.protocol, allocator.TakePalettes()));
  return result;
}

}  // namespace pypto::ir::sync
