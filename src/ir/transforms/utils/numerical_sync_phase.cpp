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

#include "src/ir/transforms/utils/numerical_sync_phase.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/compact_sync_native.h"

namespace pypto::ir::sync {
namespace {
using Ordinal = NumericalOrdinal;
using Node = NumericalNode;
using NodePtr = std::shared_ptr<const Node>;
int64_t PhaseScope(const SyncSegment& segment) {
  if (!segment.original_loop) return 1;
  return segment.nested_frames.empty() ? segment.count : segment.nested_frames.back().count;
}
int64_t Mod(Ordinal value, int64_t modulus) {
  const auto residue = value % modulus;
  return static_cast<int64_t>(residue < 0 ? residue + modulus : residue);
}
using PhaseBindings = std::map<const ForStmt*, int64_t>;
int64_t PhaseInteger(Ordinal value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max()) << "Phase coordinate overflow";
  return static_cast<int64_t>(value);
}
// One provider denotes a supplied finite microbody, not the repeated K domain.
// Enumerating its finite inner stages is charged explicitly. No source IR is
// cloned and no payload is unrolled in the executable program.
void SnapshotPhase(const SyncRegion& source, PhaseBindings& bindings, NumericalFiniteProvider& provider,
                   const CompactSyncView& result, const NumericalSyncHierarchy& hierarchy,
                   bool body_only = false, const ForStmt* periodic_frame = nullptr) {
  if (source.leaf) {
    const auto r = *source.leaf;
    const auto& segment = result.State().input.segments[r];
    const auto count = PhaseScope(segment);
    for (int64_t i = 0; i < count; ++i) {
      if (segment.original_loop) bindings[segment.original_loop.get()] = i;
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        auto site = segment.sites[s];
        bool first = true;
        bool active = !site.initialization_scope;
        for (const auto& frame : segment.nested_frames) {
          active |= frame.loop == site.initialization_scope;
          if (site.continuation && active &&
              (site.initialization_scope || frame.loop == segment.original_loop)) {
            const auto found = bindings.find(frame.loop.get());
            CHECK(found != bindings.end()) << "Phase startup depends on an unbound enclosing invocation";
            first &= found->second == 0;
          }
        }
        if (site.continuation) {
          if (!first) site.original = site.continuation;
          site.initializes_accumulator &= first;
        }
        site.continuation.reset();
        std::vector<SyncAccess> accesses;
        for (auto access : site.accesses) {
          if (hierarchy.phase_readonly_domains.count(access.storage) ||
              hierarchy.phase_owned_outputs.count(access.storage))
            continue;
          if (access.after_first && first) continue;
          access.after_first = false;
          int64_t coordinate = 0;
          if (access.row_step || access.column_step) {
            const auto found = bindings.find(access.coordinate_frame.get());
            CHECK(access.coordinate_frame && found != bindings.end() &&
                  access.coordinate_frame.get() != periodic_frame)
                << "Phase-local geometry varies outside the supplied finite microbody";
            coordinate = found->second;
            access.physical_interval.reset();
          }
          access.row = PhaseInteger(Ordinal(access.row) + Ordinal(coordinate) * access.row_step);
          access.column = PhaseInteger(Ordinal(access.column) + Ordinal(coordinate) * access.column_step);
          access.row_step = access.column_step = 0;
          access.coordinate_frame.reset();
          int64_t slot = 0;
          if (access.slot_step) {
            const auto found = bindings.find(access.slot_coordinate_frame.get());
            CHECK(access.slot_coordinate_frame && found != bindings.end())
                << "Phase slot depends on an unbound enclosing invocation";
            slot = found->second;
          }
          access.slot_first = Mod(Ordinal(access.slot_first) + Ordinal(slot) * access.slot_step,
                                  result.State().input.storage[access.storage].slot_count);
          access.slot_step = 0;
          access.slot_coordinate_frame.reset();
          accesses.push_back(std::move(access));
        }
        site.accesses = std::move(accesses);
        provider.sites.push_back(std::move(site));
      }
    }
    if (segment.original_loop) bindings.erase(segment.original_loop.get());
    return;
  }
  CHECK(body_only || !source.repetition) << "Nested repeated microbody requires its retained child interface";
  for (const auto& child : source.children)
    SnapshotPhase(child, bindings, provider, result, hierarchy, false, periodic_frame);
}
using PhasePeriods = std::map<const ForStmt*, int64_t>;
PhasePeriods IndexPhasePeriods(const CompactSyncView& result) {
  PhasePeriods periods;
  for (const auto& segment : result.State().input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        if (!access.slot_coordinate_frame || !access.slot_step) continue;
        const auto slots = result.State().input.storage[access.storage].slot_count;
        const auto orbit = slots / std::gcd(slots, access.slot_step);
        auto [position, inserted] = periods.emplace(access.slot_coordinate_frame.get(), orbit);
        if (!inserted)
          position->second =
              PhaseInteger(Ordinal(position->second / std::gcd(position->second, orbit)) * orbit);
      }
  return periods;
}
size_t RetainPhase(const SyncRegion& source, PhaseBindings bindings, bool body_only,
                   CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  const size_t id = result.State().input.segments.size() + hierarchy.finite_providers.size();
  auto& provider = hierarchy.finite_providers[id];
  SnapshotPhase(source, bindings, provider, result, hierarchy, body_only,
                body_only && source.repetition ? source.repetition->loop.get() : nullptr);
  result.Diagnostics().finite_microbody_occurrences += provider.sites.size();
  ++result.Diagnostics().phase_variants;
  return id;
}
void PlanPhases(const SyncRegion& source, CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy,
                const PhasePeriods& periods) {
  if (source.leaf) {
    CHECK(PhaseScope(result.State().input.segments[*source.leaf]) <= 1)
        << "A separate repeated child needs its retained interface outside the finite phase microbody";
    hierarchy.finite_source_providers[*source.leaf] = RetainPhase(source, {}, false, result, hierarchy);
    return;
  }
  const auto found = source.repetition ? periods.find(source.repetition->loop.get()) : periods.end();
  const auto period = found == periods.end() ? 1 : found->second;
  if (source.repetition && period > 1) {
    auto& schedule = hierarchy.phase_schedules[source.repetition->loop.get()];
    schedule.period = period;
    auto count = source.repetition->count;
    // Startup is a separate original occurrence, not a sampled recurring phase.
    // In particular phase zero later in the loop must retain accumulator reads.
    schedule.prefix_count = source.startup && count ? 1 : 0;
    PhaseBindings bindings;
    if (schedule.prefix_count) {
      bindings[source.repetition->loop.get()] = 0;
      schedule.prefix_providers.push_back(RetainPhase(source, bindings, true, result, hierarchy));
    }
    count -= schedule.prefix_count;
    schedule.cycles = count / period;
    schedule.tail_count = count % period;
    if (schedule.cycles) {
      for (int64_t phase = 0; phase < period; ++phase) {
        bindings[source.repetition->loop.get()] = schedule.prefix_count + phase;
        schedule.cycle_providers.push_back(RetainPhase(source, bindings, true, result, hierarchy));
      }
    }
    for (int64_t phase = 0; phase < schedule.tail_count; ++phase) {
      // Tail uses the same local storage phase, while its original ordinal is
      // supplied by the numerical source layout at query/export time.
      bindings[source.repetition->loop.get()] = schedule.prefix_count + phase;
      // This is exactly the same representative coordinate and startup state
      // as the corresponding retained cycle provider, not merely a slot match.
      schedule.tail_providers.push_back(schedule.cycles
                                            ? schedule.cycle_providers[phase]
                                            : RetainPhase(source, bindings, true, result, hierarchy));
    }
    return;
  }
  CHECK(!source.startup)
      << "An outer startup scope needs its own qualified phase or invariant child interface";
  for (const auto& child : source.children) PlanPhases(child, result, hierarchy, periods);
}
void PhaseCells(CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  struct Cuts {
    std::set<int64_t> rows, columns, slots;
  };
  std::map<size_t, Cuts> inventory;
  for (const auto& [id, provider] : hierarchy.finite_providers)
    for (const auto& site : provider.sites)
      for (const auto& access : site.accesses) {
        ++result.Diagnostics().interface_access_visits;
        auto& cuts = inventory[access.storage];
        cuts.rows.insert(access.row);
        cuts.rows.insert(PhaseInteger(Ordinal(access.row) + access.rows));
        cuts.columns.insert(access.column);
        cuts.columns.insert(PhaseInteger(Ordinal(access.column) + access.columns));
        cuts.slots.insert(access.slot_first);
      }
  for (const auto& [storage, cuts] : inventory) {
    std::vector<int64_t> rows(cuts.rows.begin(), cuts.rows.end()),
        columns(cuts.columns.begin(), cuts.columns.end());
    for (const auto slot : cuts.slots)
      for (size_t r = 1; r < rows.size(); ++r)
        for (size_t c = 1; c < columns.size(); ++c) {
          hierarchy.cells.push_back({storage, slot, rows[r - 1], columns[c - 1]});
          result.State().storage_classes.push_back(
              {storage, rows[r - 1], columns[c - 1], rows[r] - rows[r - 1], columns[c] - columns[c - 1]});
        }
  }
  result.Diagnostics().cells = result.Diagnostics().interface_cells = hierarchy.cells.size();
}
NodePtr BuildFinitePhase(size_t id, CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  auto& provider = hierarchy.finite_providers.at(id);
  FiniteSyncInput model;
  model.exact_accesses = true;
  for (size_t c = 0; c < hierarchy.cells.size(); ++c) model.covered_cells.push_back(c);
  for (const auto& site : provider.sites) {
    FiniteOccurrence occurrence{site.pipe, true, {}, {}};
    for (const auto& access : site.accesses) {
      model.exact_accesses &= access.exact_footprint;
      for (size_t c = 0; c < hierarchy.cells.size(); ++c) {
        ++result.Diagnostics().class_tests;
        const auto& cell = hierarchy.cells[c];
        if (cell.storage == access.storage && cell.slot == access.slot_first && cell.row >= access.row &&
            cell.row < access.row + access.rows && cell.column >= access.column &&
            cell.column < access.column + access.columns)
          (access.write ? occurrence.writes : occurrence.reads).push_back(c);
      }
    }
    model.occurrences.push_back(std::move(occurrence));
  }
  // The existing native predicate sees actual finite reference order, not
  // concatenated source-leaf inventories from a nested loop.
  CompactSyncInput native;
  native.storage = result.State().input.storage;
  native.segments.emplace_back();
  native.segments.front().sites = provider.sites;
  AccumulatorProtection protection(native);
  model.protects_storage = [&](size_t a, size_t b, Cell c) {
    return protection.Protects({0, 0, a}, {0, 0, b}, hierarchy.cells[c].storage);
  };
  provider.order = AnalyzeFiniteSync(model);
  result.Diagnostics().analyzed_occurrences += provider.sites.size();
  auto node = std::make_shared<Node>();
  node->leaf = id;
  node->length = provider.sites.size();
  for (size_t c = 0; c < hierarchy.cells.size(); ++c) {
    const auto boundary = provider.order.Boundary(c);
    INTERNAL_CHECK(boundary) << "Finite phase omitted a supplied storage cell";
    NumericalBoundary out;
    out.first_writer = boundary->first_writer;
    out.last_writer = boundary->last_writer;
    for (size_t p = 0; p < kPipeCount; ++p) {
      out.first_readers[p] = boundary->first_readers[p];
      out.last_readers[p] = boundary->last_readers[p];
    }
    node->storage.push_back(out);
  }
  for (size_t s = 0; s < provider.sites.size(); ++s) {
    const auto p = provider.sites[s].pipe;
    node->native_sites[p].push_back(s);
    if (!node->first[p]) node->first[p] = s;
    node->last[p] = s;
  }
  const auto matrix = static_cast<size_t>(PipeType::M);
  if (node->first[matrix]) {
    const auto& first = provider.sites[static_cast<size_t>(*node->first[matrix])];
    if (!first.initializes_accumulator) node->first_m_continuation = first.protected_accumulator;
    node->last_m_qualified = provider.sites[static_cast<size_t>(*node->last[matrix])].protected_accumulator;
  }
  for (const auto& event : NumericalPorts(*node, result.Diagnostics().port_normalization_visits)) {
    const auto chain =
        2 * static_cast<size_t>(provider.sites[static_cast<size_t>(event.first)].pipe) + event.second;
    node->chains[chain].push_back(event.first);
  }
  IndexNumericalPositions(*node);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    node->forward[c].resize(node->chains[c].size());
    for (size_t d = 0; d < kNumericalChains; ++d) {
      size_t target = 0;
      for (size_t source = 0; source < node->chains[c].size(); ++source) {
        while (target < node->chains[d].size()) {
          ++result.Diagnostics().interface_queries;
          if (node->chains[c][source] == node->chains[d][target] && c % 2 == d % 2) break;
          if (provider.order.Precedes(static_cast<size_t>(node->chains[c][source]), c % 2,
                                      static_cast<size_t>(node->chains[d][target]), d % 2))
            break;
          ++target;
        }
        node->forward[c][source][d] = target;
      }
    }
  }
  InvertNumericalRows(*node);
  CountNumericalInterface(*node, result);
  return node;
}
}  // namespace
void PlanNumericalPhases(const SyncRegion& source, CompactSyncBuilder& result,
                         NumericalSyncHierarchy& hierarchy) {
  PlanPhases(source, result, hierarchy, IndexPhasePeriods(result));
}
void BuildNumericalPhaseCells(CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  PhaseCells(result, hierarchy);
}
std::map<size_t, std::shared_ptr<const NumericalNode>> BuildNumericalPhaseProviders(
    CompactSyncBuilder& result, NumericalSyncHierarchy& hierarchy) {
  std::map<size_t, std::shared_ptr<const NumericalNode>> providers;
  for (const auto& [id, provider] : hierarchy.finite_providers)
    providers.emplace(id, BuildFinitePhase(id, result, hierarchy));
  return providers;
}
}  // namespace pypto::ir::sync
