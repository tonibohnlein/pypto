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

// Compose the exact finite physical tapes around a runtime work loop. The
// fallback remains symbolic in the finite trip counts. Only bounded tapes
// participate in the selection of smaller boundary sequences.
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/peer_local_work_loop.h"
#include "src/ir/transforms/utils/physical_sync_replay.h"

namespace pypto::ir::sync {
namespace {

constexpr size_t kMaxFiniteSteps = 8192;
using replay::IncludeRanks;
using replay::QueryQueues;
using replay::Ranks;
using replay::Step;
using replay::StepKind;
using DirectionList = std::vector<SyncDirection>;

struct Effect {
  int64_t begin, end;
  size_t payload;
  PipeType pipe;
  bool write;
};

struct PhaseTape {
  std::vector<Step> steps;
  std::map<std::pair<size_t, int64_t>, std::vector<Effect>> effects;

  void AddPayload(const CompactSyncInput& input, const PeerLocalWorkLoop& loop, const SyncSite& site,
                  int64_t visit) {
    steps.push_back({StepKind::Payload, {site.pipe, site.pipe}});
    for (const auto& access : site.accesses) {
      const auto [modulus, begin, end] = ProjectPeerWorkAccess(input, loop, access);
      const auto residue = static_cast<int64_t>(
          (static_cast<__int128>(visit) * access.slot_step + access.slot_first) % modulus);
      effects[{access.storage, residue}].push_back({begin, end, steps.size(), site.pipe, access.write});
    }
  }
};

// Bound the total work *before* constructing tapes. No analysis step expands
// the outer runtime loop or iterates a finite child with no effectful sites.
bool WithinBudget(const CompactSyncInput& input, const PeerLocalRegion& region,
                  const std::vector<const PhysicalSyncFamily*>& families) {
  size_t remaining = kMaxFiniteSteps;
  const auto admit = [&remaining](int64_t count, size_t cost) {
    if (cost && static_cast<uint64_t>(count) > remaining / cost) {
      return false;
    }
    remaining -= static_cast<size_t>(count) * cost;
    return true;
  };
  for (size_t r = region.first; r < region.end; ++r) {
    const auto& segment = input.segments[r];
    if (!segment.nested_frames.empty() || segment.reset_period) {
      return false;
    }
    for (const auto& site : segment.sites) {
      if (site.continuation || !admit(segment.count, 1) || !admit(segment.count, site.accesses.size())) {
        return false;
      }
      for (const auto& access : site.accesses) {
        if (access.slot_step && access.slot_coordinate_frame != segment.original_loop) {
          return false;
        }
      }
    }
  }
  for (const auto* family : families) {
    if (!family->axes.empty() || family->source_domain || family->target_domain || !admit(family->count, 2)) {
      return false;
    }
  }
  return true;
}

using MicroSteps = std::map<std::pair<Occurrence, StepKind>, std::vector<Step>>;

MicroSteps BuildMicroSteps(const CompactSyncInput& input,
                           const std::vector<const PhysicalSyncFamily*>& families) {
  MicroSteps steps;
  for (const auto* family : families) {
    const auto direction = SyncFamilyPipes(input, *family);
    for (int64_t v = 0; v < family->count; ++v) {
      const Occurrence target{family->target_region, family->target_first + v, family->target_site};
      if (family->ids.empty()) {
        steps[{target, StepKind::Barrier}].push_back({StepKind::Barrier, direction});
      } else {
        const auto id = family->ids[static_cast<size_t>(v) % family->ids.size()];
        const Occurrence source{family->source_region, family->source_first + v, family->source_site};
        steps[{source, StepKind::Set}].push_back({StepKind::Set, direction, id});
        steps[{target, StepKind::Wait}].push_back({StepKind::Wait, direction, id});
      }
    }
  }
  return steps;
}

PhaseTape BuildTape(const CompactSyncInput& input, const PeerLocalWorkLoop& loop, PeerWorkRange range,
                    const MicroSteps& commands) {
  PhaseTape tape;
  const auto append = [&tape, &commands](const Occurrence& cut, StepKind kind) {
    if (const auto at = commands.find({cut, kind}); at != commands.end()) {
      tape.steps.insert(tape.steps.end(), at->second.begin(), at->second.end());
    }
  };
  for (size_t r = range.first; r < range.second; ++r) {
    const auto& segment = input.segments[r];
    if (segment.sites.empty()) {
      continue;
    }
    for (int64_t visit = 0; visit < segment.count; ++visit) {
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        const Occurrence cut{r, visit, s};
        // Guarded direct schedule: base barriers, flag waits, original
        // issue, then flag sets. Within each group use assignment order.
        append(cut, StepKind::Barrier);
        append(cut, StepKind::Wait);
        tape.AddPayload(input, loop, segment.sites[s], visit);
        append(cut, StepKind::Set);
      }
    }
  }
  return tape;
}

// Per-queue latest overlapping producer completion required at each target
// issue. Later completions on the same FIFO imply every older completion, so
// retaining just the maximum also subsumes a whole array of byte overlaps.
using Demands = std::vector<Ranks>;

class OverlapRanks {
 public:
  explicit OverlapRanks(const std::vector<Effect>& sources) {
    for (const auto& source : sources) {
      ends_.push_back(source.end);
    }
    std::sort(ends_.begin(), ends_.end());
    ends_.erase(std::unique(ends_.begin(), ends_.end()), ends_.end());
    // Fenwick prefix in decreasing end order. Each source is inserted once
    // when its start is below the current target's increasing end.
    ranks_.resize(ends_.size() + 1);
  }

  void Insert(const Effect& source) {
    size_t at = ends_.end() - std::lower_bound(ends_.begin(), ends_.end(), source.end);
    for (; at < ranks_.size(); at += LowBit(at)) {
      auto& rank = ranks_[at][source.write][source.pipe];
      rank = std::max(rank, source.payload);
    }
  }

  void Query(const Effect& target, Ranks& required) const {
    size_t at = ends_.end() - std::upper_bound(ends_.begin(), ends_.end(), target.begin);
    for (; at; at -= LowBit(at)) {
      IncludeRanks(required, ranks_[at][true]);
      if (target.write) {
        IncludeRanks(required, ranks_[at][false]);
      }
    }
  }

 private:
  static size_t LowBit(size_t value) { return value & (size_t{0} - value); }

  std::vector<int64_t> ends_;
  std::vector<std::array<Ranks, 2>> ranks_;
};

void AddSlotDemands(const std::vector<Effect>& sources, const std::vector<Effect>& targets,
                    Demands& demands) {
  OverlapRanks index(sources);
  auto source = sources.begin();
  for (const auto& target : targets) {
    while (source != sources.end() && source->begin < target.end) {
      index.Insert(*source++);
    }
    index.Query(target, demands[target.payload - 1]);
  }
}

void SortEffects(PhaseTape& tape) {
  for (auto& [slot, effects] : tape.effects) {
    std::sort(effects.begin(), effects.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.begin < rhs.begin; });
  }
}

Demands CrossingDemands(const PhaseTape& earlier, const PhaseTape& later) {
  Demands demands(later.steps.size());
  for (const auto& [slot, targets] : later.effects) {
    const auto sources = earlier.effects.find(slot);
    if (sources == earlier.effects.end()) {
      continue;
    }
    auto by_end = targets;
    std::sort(by_end.begin(), by_end.end(),
              [](const auto& lhs, const auto& rhs) { return lhs.end < rhs.end; });
    AddSlotDemands(sources->second, by_end, demands);
  }
  return demands;
}

bool RunTape(QueryQueues& queues, const PhaseTape& tape, const Demands* required = nullptr, size_t base = 0,
             const Demands* older = nullptr, size_t older_base = 0) {
  for (size_t i = 0; i < tape.steps.size(); ++i) {
    if (!queues.Apply(tape.steps[i], required ? &(*required)[i] : nullptr, base,
                      older ? &(*older)[i] : nullptr, older_base)) {
      return false;
    }
  }
  return true;
}

bool RunBoundary(QueryQueues& queues, const DirectionList& boundary, const PeerWorkSeamIds& ids) {
  for (const auto& direction : boundary) {
    if (direction.first == direction.second) {
      if (!queues.Apply({StepKind::Barrier, direction})) {
        return false;
      }
    } else {
      const auto id = ids.at(direction);
      if (!queues.Apply({StepKind::Set, direction, id}) || !queues.Apply({StepKind::Wait, direction, id})) {
        return false;
      }
    }
  }
  return true;
}

class WorkCompletion {
 public:
  WorkCompletion(PhaseTape prefix, PhaseTape item, PhaseTape suffix, const PeerWorkSeamIds& ids,
                 bool scalar_serial)
      : prefix_(std::move(prefix)),
        item_(std::move(item)),
        suffix_(std::move(suffix)),
        enter_(CrossingDemands(prefix_, item_)),
        bypass_(CrossingDemands(prefix_, suffix_)),
        repeat_(CrossingDemands(item_, item_)),
        leave_(CrossingDemands(item_, suffix_)),
        ids_(ids),
        scalar_serial_(scalar_serial) {}

  bool Covers(const DirectionList& entry, const DirectionList& exit) const {
    QueryQueues queues(scalar_serial_);
    if (!RunTape(queues, prefix_) || !RunBoundary(queues, entry, ids_) || !queues.AllConsumed()) {
      return false;
    }
    // Check the zero-trip suffix as well as the exits after one and two
    // items. Each phase preserves old facts on a pipe. A later execution's
    // completion on that pipe also covers all earlier executions' payloads.
    // The residue projection covers the relative FIFO rotation of any two
    // items, including a prefix effect that survives a disjoint item.
    if (!Finish(queues, nullptr, 0)) {
      return false;
    }
    auto previous_item = queues.Cut();
    for (size_t i = 0; i < 2; ++i) {
      const auto item_base = queues.Cut();
      const auto first = i == 0;
      if (!RunTape(queues, item_, first ? nullptr : &repeat_, previous_item, &enter_, 0) ||
          !RunBoundary(queues, exit, ids_) || !queues.AllConsumed() || !Finish(queues, &leave_, item_base)) {
        return false;
      }
      previous_item = item_base;
    }
    // All event streams are balanced within each executed phase. A bank used
    // in the item (including its boundary) has the same previous-WAIT and
    // next-SET cuts for every repetition. A bank unused by the item retains
    // its prefix WAIT into the suffix. Additional items preserve that proof.
    return true;
  }

 private:
  bool Finish(QueryQueues queues, const Demands* leave, size_t item_base) const {
    return RunTape(queues, suffix_, leave, item_base, &bypass_, 0) && queues.AllConsumed();
  }

  const PhaseTape prefix_, item_, suffix_;
  const Demands enter_, bypass_, repeat_, leave_;
  const PeerWorkSeamIds& ids_;
  const bool scalar_serial_;
};

// Try zero, one and two services first. The fixed pipe inventory bounds this
// search independently of IR size. Larger cuts also get a linear deletion
// sweep. All candidates retain the allocated order and introduce no bank.
template <typename Accept>
DirectionList SelectBoundary(const DirectionList& original, const Accept& accept) {
  if (accept(DirectionList{})) {
    return {};
  }
  for (const auto& command : original) {
    DirectionList candidate{command};
    if (accept(candidate)) {
      return candidate;
    }
  }
  for (size_t i = 0; i < original.size(); ++i) {
    for (size_t j = i + 1; j < original.size(); ++j) {
      DirectionList candidate{original[i], original[j]};
      if (accept(candidate)) {
        return candidate;
      }
    }
  }

  auto current = original;
  for (size_t at = current.size(); at > 0; --at) {
    auto candidate = current;
    candidate.erase(candidate.begin() + at - 1);
    if (accept(candidate)) {
      current = std::move(candidate);
    }
  }
  return current;
}

}  // namespace

void SelectPeerWorkSeams(const CompactSyncInput& input, const PeerLocalRegion& region,
                         const std::vector<const PhysicalSyncFamily*>& families, const PeerWorkSeamIds& ids,
                         bool scalar_serial, PeerWorkPlacement& placement) {
  if ((placement.entry.empty() && placement.exit.empty()) || !WithinBudget(input, region, families)) {
    return;
  }
  const auto& work = *placement.loop;
  const auto commands = BuildMicroSteps(input, families);
  auto prefix = BuildTape(input, work, {region.first, work.first}, commands);
  auto item = BuildTape(input, work, {work.first, work.end}, commands);
  auto suffix = BuildTape(input, work, {work.end, region.end}, commands);
  SortEffects(prefix);
  SortEffects(item);
  SortEffects(suffix);
  const WorkCompletion proof(std::move(prefix), std::move(item), std::move(suffix), ids, scalar_serial);
  // The proof models the guarded finite assignment only. Leave the original
  // closure in place when its command semantics cannot be replayed here.
  if (!proof.Covers(placement.entry, placement.exit)) {
    return;
  }
  placement.entry = SelectBoundary(placement.entry, [&](const DirectionList& candidate) {
    return proof.Covers(candidate, placement.exit);
  });
  placement.exit = SelectBoundary(placement.exit, [&](const DirectionList& candidate) {
    return proof.Covers(placement.entry, candidate);
  });
}

}  // namespace pypto::ir::sync
