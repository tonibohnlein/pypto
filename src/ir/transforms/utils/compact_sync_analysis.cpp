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

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

#include <algorithm>
#include <chrono>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/compact_sync_native.h"
#include "src/ir/transforms/utils/numerical_sync_hierarchy.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
#include "src/ir/transforms/utils/sync_boundary_graph.h"
namespace pypto::ir::sync {

namespace {
using Atom = SyncStorageClass;
bool Intersects(const SyncAccess& a, const Atom& c, int64_t i) {
  const auto r = __int128(a.row) + __int128(i) * a.row_step;
  const auto s = __int128(a.column) + __int128(i) * a.column_step;
  return r < c.row + c.rows && c.row < r + a.rows && s < c.column + c.columns && c.column < s + a.columns;
}
bool Contains(const SyncAccess& a, const Atom& c) {
  return a.row <= c.row && a.column <= c.column && a.row + a.rows >= c.row + c.rows &&
         a.column + a.columns >= c.column + c.columns;
}
std::vector<Atom> Partition(const CompactSyncInput& input) {
  std::vector<Atom> atoms;
  std::vector<std::vector<const SyncAccess*>> writes(input.storage.size());
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses)
        if (access.write) writes.at(access.storage).push_back(&access);
  for (size_t s = 0; s < input.storage.size(); ++s) {
    const auto& storage = input.storage[s];
    if (!storage.boundary_available) continue;
    // Complete row-sweep partition: only writers active in a row band split
    // its columns. Keep holes and read-only domains covered. Global Cartesian
    // cuts would create quadratic empty cells even for disjoint diagonal writes.
    // W write rectangles cost O(W log W + C) for C emitted cells. General
    // overlapping rectangles can still have quadratic output; this is not an
    // interval-only linear-size claim.
    std::map<int64_t, std::vector<std::pair<int64_t, int>>> events;
    events[0];
    events[storage.rows];
    for (const auto* access : writes[s]) {
      const auto& a = *access;
      CHECK(a.row_step == 0 && a.column_step == 0)
          << "Moving writes need a rotating/visit-owned storage certificate";
      if (!a.rows || !a.columns) continue;
      for (int64_t column : {a.column, a.column + a.columns}) {
        events[a.row].emplace_back(column, 1);
        events[a.row + a.rows].emplace_back(column, -1);
      }
    }
    std::map<int64_t, size_t> columns{{0, 1}, {storage.columns, 1}};
    for (auto row = events.begin(); row != events.end(); ++row) {
      // Multiplicity matters when several writes share a boundary. Applying
      // both additions and removals before the next band gives half-open cells.
      for (auto [column, delta] : row->second) {
        if (delta > 0) {
          ++columns[column];
        } else {
          auto found = columns.find(column);
          INTERNAL_CHECK(found != columns.end() && found->second) << "Unbalanced storage sweep";
          if (!--found->second) columns.erase(found);
        }
      }
      auto next_row = std::next(row);
      if (next_row == events.end()) break;
      for (auto column = columns.begin(), next = std::next(column); next != columns.end(); ++column, ++next)
        atoms.push_back(
            {s, row->first, column->first, next_row->first - row->first, next->first - column->first});
    }
  }
  return atoms;
}
// One immutable directory per analysis, shared by every leaf. Cells are
// contiguous within each row band, and columns are ordered and disjoint.
struct RowBand {
  int64_t first, end;
  size_t begin_cell, end_cell;
};
using StorageIndex = std::vector<std::vector<RowBand>>;
StorageIndex IndexClasses(const CompactSyncInput& input, const std::vector<Atom>& atoms) {
  StorageIndex index(input.storage.size());
  for (size_t c = 0; c < atoms.size(); ++c) {
    const auto& atom = atoms[c];
    auto& bands = index.at(atom.storage);
    if (bands.empty() || bands.back().first != atom.row)
      bands.push_back({atom.row, atom.row + atom.rows, c, c + 1});
    else
      bands.back().end_cell = c + 1;
  }
  return index;
}
CompactSegmentResult Reduce(const SyncSegment& segment, size_t region,
                            const AccumulatorProtection& protection, const std::vector<Atom>& atoms,
                            const StorageIndex& index, size_t& inspected, size_t& class_tests,
                            size_t& class_lookup_tests) {
  const auto scope_count = segment.reset_period ? segment.reset_period : segment.count;
  FiniteSyncInput model;
  model.exact_accesses = true;
  for (size_t c = 0; c < atoms.size(); ++c) model.covered_cells.push_back(c);
  for (const auto& site : segment.sites) {
    FiniteOccurrence occurrence{site.pipe, true, {}, {}};
    for (const auto& a : site.accesses) {
      model.exact_accesses &= a.exact_footprint;
      int64_t access_count = scope_count;
      if (a.coordinate_frame)
        for (const auto& frame : segment.nested_frames)
          if (frame.loop == a.coordinate_frame) access_count = frame.count;
      const auto& bands = index.at(a.storage);
      // Query the full affine envelope, not just visit zero. Envelope lookup
      // only filters candidates: the exact all-visits proof below still refuses
      // changing membership. Wide arithmetic preserves admitted coordinates.
      const auto last_visit = std::max<int64_t>(0, access_count - 1);
      const __int128 row_end = __int128(a.row) + __int128(last_visit) * a.row_step + a.rows;
      const __int128 column_end = __int128(a.column) + __int128(last_visit) * a.column_step + a.columns;
      auto first_band = bands.begin();
      // One band covers the whole row domain; the common Vec byte domain
      // needs no row search.
      if (bands.size() > 1) {
        first_band =
            std::lower_bound(bands.begin(), bands.end(), a.row, [&](const RowBand& band, int64_t row) {
              ++class_lookup_tests;
              return band.end <= row;
            });
      }
      for (auto band = first_band; band != bands.end() && band->first < row_end; ++band) {
        auto begin = atoms.begin() + band->begin_cell, end = atoms.begin() + band->end_cell;
        begin = std::lower_bound(begin, end, a.column, [&](const Atom& atom, int64_t column) {
          ++class_lookup_tests;
          return atom.column + atom.columns <= column;
        });
        end = std::lower_bound(begin, end, column_end, [&](const Atom& atom, __int128 column) {
          ++class_lookup_tests;
          return atom.column < column;
        });
        for (auto entry = begin; entry != end; ++entry) {
          const auto c = static_cast<size_t>(entry - atoms.begin());
          ++class_tests;
          const auto& atom = atoms[c];
          if (atom.storage != a.storage) continue;
          bool first = Intersects(a, atom, 0),
               last = Intersects(a, atom, std::max<int64_t>(0, access_count - 1));
          // Nonnegative affine coordinates yield a single interval of intersecting
          // visits. Check the full interval, not just equal endpoint samples.
          int64_t lo = 0, hi = std::max<int64_t>(0, access_count - 1);
          auto intersect_axis = [&](int64_t x, int64_t extent, int64_t origin, int64_t stride, int64_t size) {
            const __int128 lower = __int128(x) - size + 1 - origin;
            const __int128 upper = __int128(x) + extent - 1 - origin;
            if (!stride) {
              if (lower > 0 || upper < 0) hi = -1;
              return;
            }
            if (upper < 0) {
              hi = -1;
              return;
            }
            hi = static_cast<int64_t>(std::min<__int128>(hi, upper / stride));
            if (lower > 0) {
              const auto minimum = (lower + stride - 1) / stride;
              if (minimum > hi)
                hi = -1;
              else
                lo = std::max(lo, static_cast<int64_t>(minimum));
            }
          };
          intersect_axis(atom.row, atom.rows, a.row, a.row_step, a.rows);
          intersect_axis(atom.column, atom.columns, a.column, a.column_step, a.columns);
          const bool any = lo <= hi;
          CHECK((first && last) || !any)
              << "Storage membership varies across visits; selector partition unavailable";
          if (!first) continue;
          CHECK(!a.write || Contains(a, atom)) << "Write class is only partially covered";
          (a.write ? occurrence.writes : occurrence.reads).push_back(c);
        }
      }
    }
    model.occurrences.push_back(std::move(occurrence));
  }
  const size_t sites = segment.sites.size();
  model.protects_storage = [&](size_t a, size_t b, Cell cell) {
    return protection.Protects({region, static_cast<int64_t>(a / sites), a % sites},
                               {region, static_cast<int64_t>(b / sites), b % sites}, atoms[cell].storage);
  };
  CompactSegmentResult result;
  result.count = segment.count;
  result.internal_scope_count = segment.original_loop ? scope_count : 1;
  result.body = AnalyzeFiniteSync(model);
  const size_t m = model.occurrences.size();
  for (const auto& site : segment.sites) result.pipes.push_back(site.pipe);
  if (!segment.original_loop) {
    for (auto [a, b] : result.body.generators) result.generators.push_back({a, b, 0});
    for (auto [a, b] : result.body.demands) result.demands.push_back({a, b, 0});
    inspected += m;
    return result;
  }
  auto two = model;
  two.occurrences.insert(two.occurrences.end(), model.occurrences.begin(), model.occurrences.end());
  auto witness = AnalyzeFiniteSync(two);
  inspected += m + 2 * m;
  auto families = [&](const std::vector<Demand>& edges) {
    std::set<std::tuple<size_t, size_t, int64_t>> unique;
    for (const auto& [a, b] : edges)
      if (a < m) unique.emplace(a % m, b % m, static_cast<int64_t>(b / m - a / m));
    std::vector<DemandFamily> out;
    for (auto [a, b, d] : unique) out.push_back({a, b, d});
    return out;
  };
  if (m) {
    result.generators = families(witness.generators);
    result.demands = families(witness.demands);
  }
  // Invariant membership, native pipe order and nonnegative wrap generators
  // establish the same periodic query contract for flat and nested leaves.
  // Retain two rows per participating pipe, not one row per event.
  result.core_frontiers = true;
  result.quotient_edges = BuildPeriodicCoreFrontiers(result);
  return result;
}
}  // namespace

std::optional<__int128> CompactSegmentResult::Threshold(size_t source, bool sc, size_t target,
                                                        bool tc) const {
  CHECK(source < pipes.size() && target < pipes.size()) << "Invalid compact event type";
  if (!rotating && !core_frontiers) {
    return (source == target && sc == tc) || body.Precedes(source, sc, target, tc) ? std::optional<int64_t>(0)
                                                                                   : std::nullopt;
  }
  const size_t pipe = static_cast<size_t>(pipes[source]);
  const auto& value = frontier_distances[pipe + (sc ? kPipeCount : 0)].at(2 * target + tc);
  if (!value) return std::nullopt;
  const int64_t h = pipe_sizes[pipe];
  // ceil((rank - beta)/h), beta=h-D. Avoid adding h to a potentially
  // large distance: this equivalent form uses quotient and remainder.
  const __int128 numerator = *value - (h - ranks[source]);
  const __int128 threshold = numerator / h + (numerator % h > 0);
  INTERNAL_CHECK(threshold >= 0) << "Periodic query threshold must be nonnegative";
  return threshold;
}
bool CompactSegmentResult::Precedes(int64_t a, size_t source, bool sc, int64_t b, size_t target,
                                    bool tc) const {
  CHECK(a >= 0 && b >= 0 && a < count && b < count && source < pipes.size() && target < pipes.size())
      << "Invalid compact synchronization occurrence";
  if (a > b || (a == b && source == target && sc == tc)) return false;
  const auto threshold = Threshold(source, sc, target, tc);
  return threshold && *threshold <= b - a;
}

void IndexSyncStorageAccesses(CompactSyncBuilder& result) {
  INTERNAL_CHECK(result.State().storage_access_index.empty())
      << "Original access index requires a fresh result";
  result.State().storage_access_index.resize(result.State().input.storage.size());
  for (size_t r = 0; r < result.State().input.segments.size(); ++r) {
    const auto& segment = result.State().input.segments[r];
    for (size_t s = 0; s < segment.sites.size(); ++s)
      for (size_t a = 0; a < segment.sites[s].accesses.size(); ++a) {
        const auto& access = segment.sites[s].accesses[a];
        auto& index = result.State().storage_access_index.at(access.storage);
        index.accesses.emplace_back(r, s, a);
        index.written |= access.write;
        index.external_geometry |= static_cast<bool>(access.external);
        index.enclosing_motion |= access.coordinate_frame && access.coordinate_frame != segment.original_loop;
      }
  }
}
CompactSyncBuilder AnalyzeInvariantSyncLeaves(CompactSyncInput input) {
  CompactSyncBuilder result;
  result.State().input = std::move(input);
  using Clock = std::chrono::steady_clock;
  auto begin = Clock::now();
  IndexSyncStorageAccesses(result);
  auto atoms = Partition(result.State().input);
  result.Diagnostics().cells = atoms.size();
  result.State().storage_classes = atoms;
  const auto index = IndexClasses(result.State().input, atoms);
  auto partitioned = Clock::now();
  result.Diagnostics().partition_seconds = std::chrono::duration<double>(partitioned - begin).count();
  AccumulatorProtection protection(result.State().input);
  for (size_t r = 0; r < result.State().input.segments.size(); ++r)
    result.State().segments.push_back(Reduce(result.State().input.segments[r], r, protection, atoms, index,
                                             result.Diagnostics().analyzed_occurrences,
                                             result.Diagnostics().class_tests,
                                             result.Diagnostics().class_lookup_tests));
  if (result.State().input.region_tree)
    for (auto& segment : result.State().segments) segment.count = segment.internal_scope_count;
  auto reduced = Clock::now();
  result.Diagnostics().reduction_seconds = std::chrono::duration<double>(reduced - partitioned).count();
  return result;
}
CompactSyncResult AnalyzeCompactSync(CompactSyncInput input, const std::string& composition_order) {
  CHECK(composition_order == "balanced" || composition_order == "left" || composition_order == "right")
      << "Unknown numerical composition order";
  const bool rotating =
      std::any_of(input.segments.begin(), input.segments.end(), [&](const SyncSegment& region) {
        return std::any_of(region.sites.begin(), region.sites.end(), [&](const SyncSite& site) {
          return std::any_of(site.accesses.begin(), site.accesses.end(), [&](const SyncAccess& access) {
            const auto& storage = input.storage.at(access.storage);
            return storage.slot_count > 1 ||
                   ((storage.byte_coordinates || storage.space == MemorySpace::DDR) && access.write &&
                    (access.row_step || access.column_step));
          });
        });
      });
  if (input.region_tree) return AnalyzeNumericalSync(std::move(input), composition_order).Publish();
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses)
        CHECK(!access.external) << "External coordinate geometry requires a qualified regional consumer";
  if (rotating)
    return input.segments.size() > 1 ||
                   (!input.segments.empty() && !input.segments.front().nested_frames.empty())
               ? AnalyzeNumericalSync(std::move(input), composition_order).Publish()
               : AnalyzeRotatingSync(std::move(input)).Publish();
  auto result = AnalyzeInvariantSyncLeaves(std::move(input));
  const auto& atoms = result.State().storage_classes;
  using Clock = std::chrono::steady_clock;
  const auto reduced = Clock::now();
  // A single region already owns all queries and boundaries. Keep the
  // concrete numerical fast path free of a redundant all-pairs port graph.
  if (result.State().segments.size() == 1) return std::move(result).Publish();
  AccumulatorProtection protection(result.State().input);
  std::vector<std::optional<Occurrence>> writers(atoms.size());
  std::vector<std::array<std::optional<Occurrence>, kPipeCount>> readers(atoms.size());
  std::set<Crossing> crossings;
  for (size_t r = 0; r < result.State().segments.size(); ++r) {
    const auto& s = result.State().segments[r];
    if (!s.count || s.pipes.empty()) continue;
    for (size_t c = 0; c < atoms.size(); ++c) {
      auto boundary = s.body.Boundary(c);
      INTERNAL_CHECK(boundary) << "Missing supplied cell";
      auto crossing = [&](const Occurrence& a, size_t target) {
        const Occurrence b{r, 0, target};
        if (!protection.Protects(a, b, atoms[c].storage)) crossings.insert({a, b});
      };
      if (writers[c]) {
        if (boundary->first_writer) crossing(*writers[c], *boundary->first_writer);
        for (auto reader : boundary->first_readers)
          if (reader) crossing(*writers[c], *reader);
      }
      if (boundary->first_writer) {
        for (auto reader : readers[c])
          if (reader) crossing(*reader, *boundary->first_writer);
        writers[c] = Occurrence{r, s.count - 1, *boundary->last_writer};
        readers[c].fill(std::nullopt);
      }
      for (size_t p = 0; p < kPipeCount; ++p)
        if (boundary->last_readers[p]) readers[c][p] = Occurrence{r, s.count - 1, *boundary->last_readers[p]};
    }
  }
  result.State().crossing_generators.assign(crossings.begin(), crossings.end());
  // Supply every generator endpoint before the immutable port index is built.
  // Reduction must query the required graph, not an already reduced subset.
  std::vector<Occurrence> ports;
  for (const auto& crossing : crossings) {
    ports.push_back(crossing.first);
    ports.push_back(crossing.second);
  }
  auto boundary = std::make_shared<SyncBoundaryGraph>(result.State().segments, ports);
  result.State().boundary_order = boundary;
  result.Diagnostics().port_pair_tests = boundary->pair_tests;
  for (const auto& e : crossings) boundary->Add({e.first, true}, {e.second, false});
  for (const auto& e : crossings) {
    ++result.Diagnostics().crossing_path_searches;
    auto a = boundary->Node({e.first, true}), b = boundary->Node({e.second, false});
    if (!boundary->Path(a, b, std::make_pair(a, b))) result.State().crossing_demands.push_back(e);
  }
  result.Diagnostics().graph_edges = boundary->Edges();
  result.Diagnostics().composition_seconds = std::chrono::duration<double>(Clock::now() - reduced).count();
  return std::move(result).Publish();
}
bool CompactSyncView::Precedes(const Occurrence& a, bool sc, const Occurrence& b, bool tc) const {
  if (State().hierarchy) return NumericalPrecedes(*this, a, sc, b, tc);
  auto [ar, av, as] = a;
  auto [br, bv, bs] = b;
  CHECK(ar < State().segments.size() && br < State().segments.size()) << "Invalid compact region";
  // Validate coordinates even for a backwards query.
  State().segments[ar].Precedes(av, as, sc, av, as, sc);
  State().segments[br].Precedes(bv, bs, tc, bv, bs, tc);
  if (ar == br) return State().segments[ar].Precedes(av, as, sc, bv, bs, tc);
  if (ar > br) return false;
  INTERNAL_CHECK(State().boundary_order) << "Missing original boundary order";
  return State().boundary_order->Precedes(State().segments, a, sc, b, tc);
}
std::optional<std::vector<Crossing>> CompactSyncView::ExportCrossingDemands() const {
  return ExportNumericalCrossings(*this);
}
std::shared_ptr<const ScopedSyncDemands> CompactSyncResult::ExportScopedDemands() const {
  if (!scoped_export_->attempted) {
    // Cache unavailable as well as successful capabilities; neither changes the
    // retained mathematics. Exceptions leave the request eligible for retry.
    ++Diagnostics().scoped_export_attempts;
    auto value = ExportNumericalScopedDemands(*this);
    scoped_export_->value = std::move(value);
    scoped_export_->attempted = true;
    if (scoped_export_->value) ++Diagnostics().scoped_export_builds;
  }
  return scoped_export_->value;
}
bool CompactSyncView::ReferenceBefore(const Occurrence& a, const Occurrence& b) const {
  return State().hierarchy ? NumericalReferenceBefore(*this, a, b) : a < b;
}
}  // namespace pypto::ir::sync
