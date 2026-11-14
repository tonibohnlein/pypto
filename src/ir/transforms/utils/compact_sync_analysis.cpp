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
#include <limits>
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
std::optional<std::pair<int64_t, int64_t>> Visits(const SyncAccess& a, int64_t count, int64_t row,
                                                  int64_t col) {
  if (count <= 0) return std::nullopt;
  int64_t lo = a.after_first ? 1 : 0, hi = count - 1;
  auto restrict = [&](int64_t x, int64_t start, int64_t step, int64_t size) {
    if (step == 0) {
      if (x < start || x - start >= size) hi = -1;
      return;
    }
    // start+i*step <= x < start+i*step+size; all offsets/steps are nonnegative.
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
// Exact first/last matching coordinates in a repeated rectangular visit box.
// Only its innermost coordinate moves the read footprint; enclosing coordinates
// repeat the same generation. Min/max outer corners preserve the original
// mixed-radix identity without treating gaps between inner ranges as accesses.
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

std::optional<std::pair<int64_t, int64_t>> OriginalAccessPointVisits(const SyncAccess& access,
                                                                     const SyncSegment& segment, int64_t row,
                                                                     int64_t column) {
  CHECK(!access.external) << "External point selector requires its qualified coordinate provider";
  return PointVisits(access, segment, row, column);
}
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

void IndexSyncStorageAccesses(CompactSyncResult& result) {
  INTERNAL_CHECK(result.storage_access_index.empty()) << "Original access index requires a fresh result";
  result.storage_access_index.resize(result.input.storage.size());
  for (size_t r = 0; r < result.input.segments.size(); ++r) {
    const auto& segment = result.input.segments[r];
    for (size_t s = 0; s < segment.sites.size(); ++s)
      for (size_t a = 0; a < segment.sites[s].accesses.size(); ++a) {
        const auto& access = segment.sites[s].accesses[a];
        auto& index = result.storage_access_index.at(access.storage);
        index.accesses.emplace_back(r, s, a);
        index.written |= access.write;
        index.external_geometry |= static_cast<bool>(access.external);
        index.enclosing_motion |= access.coordinate_frame && access.coordinate_frame != segment.original_loop;
      }
  }
}
CompactSyncResult AnalyzeInvariantSyncLeaves(CompactSyncInput input) {
  CompactSyncResult result;
  result.input = std::move(input);
  using Clock = std::chrono::steady_clock;
  auto begin = Clock::now();
  IndexSyncStorageAccesses(result);
  auto atoms = Partition(result.input);
  result.cells = atoms.size();
  result.storage_classes = atoms;
  const auto index = IndexClasses(result.input, atoms);
  auto partitioned = Clock::now();
  result.partition_seconds = std::chrono::duration<double>(partitioned - begin).count();
  AccumulatorProtection protection(result.input);
  for (size_t r = 0; r < result.input.segments.size(); ++r)
    result.segments.push_back(Reduce(result.input.segments[r], r, protection, atoms, index,
                                     result.analyzed_occurrences, result.class_tests,
                                     result.class_lookup_tests));
  if (result.input.region_tree)
    for (auto& segment : result.segments) segment.count = segment.internal_scope_count;
  auto reduced = Clock::now();
  result.reduction_seconds = std::chrono::duration<double>(reduced - partitioned).count();
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
  if (input.region_tree) return AnalyzeNumericalSync(std::move(input), composition_order);
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses)
        CHECK(!access.external) << "External coordinate geometry requires a qualified regional consumer";
  if (rotating)
    return input.segments.size() > 1 ||
                   (!input.segments.empty() && !input.segments.front().nested_frames.empty())
               ? AnalyzeNumericalSync(std::move(input), composition_order)
               : AnalyzeRotatingSync(std::move(input));
  auto result = AnalyzeInvariantSyncLeaves(std::move(input));
  const auto& atoms = result.storage_classes;
  using Clock = std::chrono::steady_clock;
  const auto reduced = Clock::now();
  // A single region already owns all queries and boundaries. Keep the
  // concrete numerical fast path free of a redundant all-pairs port graph.
  if (result.segments.size() == 1) return result;
  AccumulatorProtection protection(result.input);
  std::vector<std::optional<Occurrence>> writers(atoms.size());
  std::vector<std::array<std::optional<Occurrence>, kPipeCount>> readers(atoms.size());
  std::set<Crossing> crossings;
  for (size_t r = 0; r < result.segments.size(); ++r) {
    const auto& s = result.segments[r];
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
  result.crossing_generators.assign(crossings.begin(), crossings.end());
  // Supply every generator endpoint before the immutable port index is built.
  // Reduction must query the required graph, not an already reduced subset.
  std::vector<Occurrence> ports;
  for (const auto& crossing : crossings) {
    ports.push_back(crossing.first);
    ports.push_back(crossing.second);
  }
  result.boundary_order_ = std::make_shared<SyncBoundaryGraph>(result.segments, ports);
  result.port_pair_tests = result.boundary_order_->pair_tests;
  for (const auto& e : crossings) result.boundary_order_->Add({e.first, true}, {e.second, false});
  for (const auto& e : crossings) {
    ++result.crossing_path_searches;
    auto a = result.boundary_order_->Node({e.first, true}),
         b = result.boundary_order_->Node({e.second, false});
    if (!result.boundary_order_->Path(a, b, std::make_pair(a, b))) result.crossing_demands.push_back(e);
  }
  result.graph_edges = result.boundary_order_->Edges();
  result.composition_seconds = std::chrono::duration<double>(Clock::now() - reduced).count();
  return result;
}
bool CompactSyncResult::Precedes(const Occurrence& a, bool sc, const Occurrence& b, bool tc) const {
  if (hierarchy) return NumericalPrecedes(*this, a, sc, b, tc);
  auto [ar, av, as] = a;
  auto [br, bv, bs] = b;
  CHECK(ar < segments.size() && br < segments.size()) << "Invalid compact region";
  // Validate coordinates even for a backwards query.
  segments[ar].Precedes(av, as, sc, av, as, sc);
  segments[br].Precedes(bv, bs, tc, bv, bs, tc);
  if (ar == br) return segments[ar].Precedes(av, as, sc, bv, bs, tc);
  if (ar > br) return false;
  INTERNAL_CHECK(boundary_order_) << "Missing original boundary order";
  return boundary_order_->Precedes(segments, a, sc, b, tc);
}
std::optional<PointBoundary> CompactSyncResult::Boundary(size_t storage, int64_t row, int64_t column) const {
  return BoundaryAt(storage, row, column, {});
}
std::optional<PointBoundary> CompactSyncResult::BoundaryAt(size_t storage, int64_t row, int64_t column,
                                                           const std::map<size_t, int64_t>& parameters,
                                                           std::optional<int64_t> block_index) const {
  if (storage >= input.storage.size() || input.storage[storage].slot_count != 1) return std::nullopt;
  return BoundarySlot(storage, 0, row, column, parameters, block_index);
}
std::optional<PointBoundary> CompactSyncResult::BoundarySlot(size_t storage, int64_t slot, int64_t row,
                                                             int64_t column,
                                                             const std::map<size_t, int64_t>& parameters,
                                                             std::optional<int64_t> block_index) const {
  if (storage >= input.storage.size() || slot < 0 || slot >= input.storage[storage].slot_count)
    return std::nullopt;
  const auto& domain = input.storage[storage];
  if (domain.row_extent_parameter) {
    const auto parameter = *domain.row_extent_parameter;
    auto value = parameters.find(parameter);
    const auto bounds = input.parameter_bounds.at(parameter);
    if (value == parameters.end() || value->second < bounds.first || value->second > bounds.second ||
        row >= value->second)
      return std::nullopt;
  }
  if (row < 0 || column < 0 || row >= domain.rows || column >= domain.columns) return std::nullopt;
  if (hierarchy && storage_access_index.at(storage).external_geometry)
    return QueryExternalNumericalBoundary(*this, storage, slot, row, column, parameters, block_index);
  auto shift = [&](const SyncTranslation& translation) -> std::optional<int64_t> {
    int64_t value = 0;
    for (auto [index, coefficient] : translation) {
      auto bound = input.parameter_bounds.find(index);
      auto supplied = parameters.find(index);
      if (bound == input.parameter_bounds.end() || supplied == parameters.end() ||
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
    if (!block_index || !input.block_bounds || *block_index < input.block_bounds->first ||
        *block_index > input.block_bounds->second)
      return std::nullopt;
    const auto coordinate = *block_index - input.block_bounds->first;
    const auto row_shift = __int128(*r) + __int128(domain.row_block) * coordinate;
    const auto column_shift = __int128(*c) + __int128(domain.column_block) * coordinate;
    if (row_shift > std::numeric_limits<int64_t>::max() || column_shift > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    r = static_cast<int64_t>(row_shift);
    c = static_cast<int64_t>(column_shift);
  }
  // The original domain is covered, but this invocation accesses only its
  // translated footprint. A point before that footprint is proven absent.
  if (row < *r || column < *c) return PointBoundary{};
  if (hierarchy && input.region_tree) {
    const auto& index = storage_access_index.at(storage);
    if (index.enclosing_motion) {
      if (index.written) return std::nullopt;  // No invented absence for an unsupported selector.
      PointBoundary out;
      for (const auto& [leaf, site, access_index] : index.accesses) {
        ++point_selector_access_visits;
        const auto& segment = input.segments[leaf];
        const auto& access = segment.sites[site].accesses[access_index];
        auto visits = PointVisits(access, segment, row - *r, column - *c);
        if (!visits) continue;
        const auto pipe = static_cast<size_t>(segment.sites[site].pipe);
        const Occurrence first{leaf, visits->first, site}, last{leaf, visits->second, site};
        if (!out.first_readers[pipe] || Precedes(first, false, *out.first_readers[pipe], false))
          out.first_readers[pipe] = first;
        if (!out.last_readers[pipe] || Precedes(*out.last_readers[pipe], false, last, false))
          out.last_readers[pipe] = last;
      }
      return out;
    }
  }
  if (hierarchy) return QueryNumericalBoundary(*this, storage, slot, row - *r, column - *c);
  if (segments.size() == 1 && segments.front().rotating)
    return RotatingBoundary(input, segments.front(), storage, slot, row - *r, column - *c);
  return RelativeBoundary(storage, row - *r, column - *c);
}
std::optional<PointBoundary> CompactSyncResult::RelativeBoundary(size_t storage, int64_t row, int64_t column,
                                                                 std::optional<size_t> leaf) const {
  if (storage >= input.storage.size() || row < 0 || column < 0 || row >= input.storage[storage].rows ||
      column >= input.storage[storage].columns)
    return std::nullopt;
  PointBoundary out;
  struct Range {
    size_t r, s;
    int64_t lo, hi;
    bool write;
    PipeType pipe;
  };
  std::vector<Range> ranges;
  for (size_t r = leaf.value_or(0); r < (leaf ? *leaf + 1 : input.segments.size()); ++r)
    for (size_t s = 0; s < input.segments[r].sites.size(); ++s) {
      const auto& site = input.segments[r].sites[s];
      for (const auto& a : site.accesses)
        if (a.storage == storage) {
          if (a.external) return std::nullopt;
          auto visits = [&](const SyncAccess& access) {
            return leaf ? Visits(access, segments[r].internal_scope_count, row, column)
                        : PointVisits(access, input.segments[r], row, column);
          };
          auto v = visits(a);
          if (!v) continue;
          // A read-modify-write contributes a writer, not a read-only boundary.
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
PointBoundary CompactSyncResult::LeafBoundary(size_t region, size_t storage, int64_t row,
                                              int64_t column) const {
  CHECK(region < segments.size()) << "Invalid compact leaf";
  for (const auto& site : input.segments[region].sites)
    for (const auto& access : site.accesses)
      CHECK(access.storage != storage ||
            (!access.external &&
             (!access.coordinate_frame || access.coordinate_frame == input.segments[region].original_loop)))
          << "Leaf point selector requires its enclosing occurrence binding";
  auto result = RelativeBoundary(storage, row, column, region);
  CHECK(result) << "Leaf storage point outside the supplied domain";
  return *result;
}
std::optional<std::vector<Crossing>> CompactSyncResult::ExportCrossingDemands() const {
  return ExportNumericalCrossings(*this);
}
std::shared_ptr<const ScopedSyncDemands> CompactSyncResult::ExportScopedDemands() const {
  if (!scoped_demands_) {
    scoped_demands_ = ExportNumericalScopedDemands(*this);
    if (scoped_demands_) ++scoped_export_builds;
  }
  return scoped_demands_;
}
bool CompactSyncResult::ReferenceBefore(const Occurrence& a, const Occurrence& b) const {
  return hierarchy ? NumericalReferenceBefore(*this, a, b) : a < b;
}
std::vector<Crossing> CompactSyncResult::ExpandRetainedDemands(size_t max_payloads) const {
  return ExpandNumericalDemands(*this, max_payloads);
}
}  // namespace pypto::ir::sync
