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
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/numerical_sync_hierarchy.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
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
    std::set<int64_t> rows{0, storage.rows}, cols{0, storage.columns};
    for (const auto* access : writes[s]) {
      const auto& a = *access;
      CHECK(a.row_step == 0 && a.column_step == 0)
          << "Moving writes need a rotating/visit-owned storage certificate";
      rows.insert(a.row);
      rows.insert(a.row + a.rows);
      cols.insert(a.column);
      cols.insert(a.column + a.columns);
    }
    // Only static write boundaries split classes. Read-only changing slices
    // do not enumerate trip counts or bytes. Below we prove membership of each
    // read against every class is invariant, or refuse that abstraction.
    std::vector<int64_t> r(rows.begin(), rows.end()), c(cols.begin(), cols.end());
    for (size_t i = 1; i < r.size(); ++i)
      for (size_t j = 1; j < c.size(); ++j)
        atoms.push_back({s, r[i - 1], c[j - 1], r[i] - r[i - 1], c[j] - c[j - 1]});
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
// Sparse occurrence counters retain whole-loop multiplicities, never visits.
// Each access/site is indexed once; pair qualification costs logarithmic lookup.
// 128-bit counts cover size_t static sites times signed-64-bit trip counts.
class OccurrenceCounter {
  struct Slice {
    size_t region;
    __int128 before;
    int64_t count;
    std::vector<size_t> sites;
  };
  std::vector<Slice> slices_;

 public:
  void Add(size_t region, int64_t count, size_t site) {
    if (slices_.empty() || slices_.back().region != region) {
      __int128 before = 0;
      if (!slices_.empty()) {
        const auto& last = slices_.back();
        before = last.before + __int128(last.count) * last.sites.size();
      }
      slices_.push_back({region, before, count, {}});
    }
    auto& sites = slices_.back().sites;
    if (sites.empty() || sites.back() != site) sites.push_back(site);
  }
  __int128 Before(const Occurrence& occurrence) const {
    const auto [region, visit, site] = occurrence;
    auto it = std::upper_bound(slices_.begin(), slices_.end(), region,
                               [](size_t r, const Slice& slice) { return r < slice.region; });
    if (it == slices_.begin()) return 0;
    --it;
    if (it->region != region) return it->before + __int128(it->count) * it->sites.size();
    return it->before + __int128(visit) * it->sites.size() +
           (std::lower_bound(it->sites.begin(), it->sites.end(), site) - it->sites.begin());
  }
};
class AccumulatorProtection {
  const CompactSyncInput& input_;
  OccurrenceCounter matrix_;
  std::map<size_t, OccurrenceCounter> qualified_, interference_;

 public:
  explicit AccumulatorProtection(const CompactSyncInput& input) : input_(input) {
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      for (size_t s = 0; s < segment.sites.size(); ++s) {
        const auto& site = segment.sites[s];
        if (site.pipe == PipeType::M) matrix_.Add(r, segment.count, s);
        if (site.protected_accumulator) qualified_[*site.protected_accumulator].Add(r, segment.count, s);
        for (const auto& access : site.accesses)
          if (input.storage[access.storage].space == MemorySpace::Acc &&
              site.protected_accumulator != access.storage)
            interference_[access.storage].Add(r, segment.count, s);
      }
    }
  }
  bool Protects(const Occurrence& a, const Occurrence& b, size_t storage) const {
    const auto [ar, av, as] = a;
    const auto [br, bv, bs] = b;
    const auto& source = input_.segments[ar].sites[as];
    const auto& target = input_.segments[br].sites[bs];
    if (source.protected_accumulator != storage || target.protected_accumulator != storage ||
        (target.initializes_accumulator && (!target.continuation || bv == 0)))
      return false;
    // Every intervening M operation must be qualified on this same physical
    // accumulator. Any other access (including a FIX read) breaks protection.
    // This is NOT a C->I edge: consumers on other pipes still need synchronization.
    const auto& qualified = qualified_.at(storage);
    if (matrix_.Before(b) - matrix_.Before(a) != qualified.Before(b) - qualified.Before(a)) return false;
    auto found = interference_.find(storage);
    return found == interference_.end() || found->second.Before(a) == found->second.Before(b);
  }
};
CompactSegmentResult Reduce(const SyncSegment& segment, size_t region,
                            const AccumulatorProtection& protection, const std::vector<Atom>& atoms,
                            size_t& inspected, size_t& class_tests, size_t& class_lookup_tests,
                            bool core_frontiers) {
  const auto scope_count = segment.reset_period ? segment.reset_period : segment.count;
  FiniteSyncInput model;
  model.exact_accesses = true;
  for (size_t c = 0; c < atoms.size(); ++c) model.covered_cells.push_back(c);
  std::map<size_t, std::vector<size_t>> by_storage;
  for (size_t c = 0; c < atoms.size(); ++c) by_storage[atoms[c].storage].push_back(c);
  for (const auto& site : segment.sites) {
    FiniteOccurrence occurrence{site.pipe, true, {}, {}};
    for (const auto& a : site.accesses) {
      int64_t access_count = scope_count;
      if (a.coordinate_frame)
        for (const auto& frame : segment.nested_frames)
          if (frame.loop == a.coordinate_frame) access_count = frame.count;
      const auto& directory = by_storage.at(a.storage);
      auto begin = directory.begin(), end = directory.end();
      // Partition emits rows, then ordered disjoint column intervals. A single
      // row band (in particular the shared Vec byte domain) permits indexed
      // lookup without inspecting unrelated allocations. Keep full membership
      // qualification below; moving intervals still use the general proof.
      if (!directory.empty() && a.column_step == 0 &&
          atoms[directory.front()].row == atoms[directory.back()].row) {
        begin = std::lower_bound(begin, end, a.column, [&](size_t c, int64_t column) {
          ++class_lookup_tests;
          return atoms[c].column + atoms[c].columns <= column;
        });
        end = std::lower_bound(begin, end, a.column + a.columns, [&](size_t c, int64_t column) {
          ++class_lookup_tests;
          return atoms[c].column < column;
        });
      }
      for (auto entry = begin; entry != end; ++entry) {
        const auto c = *entry;
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
  if (core_frontiers) {
    result.core_frontiers = true;
    result.quotient_edges = BuildPeriodicCoreFrontiers(result);
    return result;
  }
  // Weighted periodic event graph. A path's weight is the visit displacement.
  // Native wrap wires allow padding a path by any number of full visits at
  // its source, hence a minimum weight answers every later compatible visit.
  std::vector<std::vector<std::pair<size_t, int64_t>>> graph(2 * m);
  std::array<std::vector<size_t>, kPipeCount> pipes;
  for (size_t s = 0; s < m; ++s) {
    graph[2 * s].emplace_back(2 * s + 1, 0);
    pipes[result.pipes[s]].push_back(s);
  }
  for (const auto& pipe : pipes)
    for (size_t i = 0; i < pipe.size(); ++i) {
      auto a = pipe[i], b = pipe[(i + 1) % pipe.size()];
      auto delta = i + 1 == pipe.size() ? 1 : 0;
      graph[2 * a].emplace_back(2 * b, delta);
      graph[2 * a + 1].emplace_back(2 * b + 1, delta);
    }
  for (auto f : result.generators) graph[2 * f.source + 1].emplace_back(2 * f.target, f.distance);
  for (size_t source = 0; source < 2 * m; ++source) {
    std::vector<int64_t> distances(2 * m, -1);
    distances[source] = 0;
    std::deque<size_t> queue{source};
    while (!queue.empty()) {
      auto a = queue.front();
      queue.pop_front();
      for (auto [b, w] : graph[a])
        if (distances[b] < 0 || distances[b] > distances[a] + w) {
          distances[b] = distances[a] + w;
          if (w)
            queue.push_back(b);
          else
            queue.push_front(b);
        }
    }
    result.distance.push_back(std::move(distances));
  }
  return result;
}
}  // namespace
std::optional<__int128> CompactSegmentResult::Threshold(size_t source, bool sc, size_t target,
                                                        bool tc) const {
  CHECK(source < pipes.size() && target < pipes.size()) << "Invalid compact event type";
  if (!rotating && !core_frontiers) {
    if (!distance.empty()) {
      const auto value = distance[2 * source + sc][2 * target + tc];
      return value < 0 ? std::nullopt : std::optional<int64_t>(value);
    }
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
size_t CompactSyncResult::Node(const Event& e) {
  auto [found, inserted] = nodes_.emplace(e, nodes_.size());
  if (inserted) graph_.emplace_back();
  return found->second;
}
bool CompactSyncResult::Path(size_t a, size_t b, std::optional<std::pair<size_t, size_t>> skip) const {
  std::vector<bool> seen(graph_.size());
  std::vector<size_t> queue{a};
  seen[a] = true;
  while (!queue.empty()) {
    auto x = queue.back();
    queue.pop_back();
    for (auto y : graph_[x]) {
      if (skip && skip->first == x && skip->second == y) continue;
      if (y == b) return true;
      if (!seen[y]) {
        seen[y] = true;
        queue.push_back(y);
      }
    }
  }
  return false;
}
CompactSyncResult AnalyzeInvariantSyncLeaves(CompactSyncInput input) {
  CompactSyncResult result;
  result.input = std::move(input);
  using Clock = std::chrono::steady_clock;
  auto begin = Clock::now();
  result.storage_access_index.resize(result.input.storage.size());
  for (size_t r = 0; r < result.input.segments.size(); ++r) {
    const auto& segment = result.input.segments[r];
    for (size_t s = 0; s < segment.sites.size(); ++s)
      for (size_t a = 0; a < segment.sites[s].accesses.size(); ++a) {
        const auto& access = segment.sites[s].accesses[a];
        auto& index = result.storage_access_index.at(access.storage);
        index.accesses.emplace_back(r, s, a);
        index.written |= access.write;
        index.enclosing_motion |= access.coordinate_frame && access.coordinate_frame != segment.original_loop;
      }
  }
  auto atoms = Partition(result.input);
  result.cells = atoms.size();
  result.storage_classes = atoms;
  auto partitioned = Clock::now();
  result.partition_seconds = std::chrono::duration<double>(partitioned - begin).count();
  AccumulatorProtection protection(result.input);
  for (size_t r = 0; r < result.input.segments.size(); ++r)
    result.segments.push_back(Reduce(result.input.segments[r], r, protection, atoms,
                                     result.analyzed_occurrences, result.class_tests,
                                     result.class_lookup_tests, result.input.region_tree.has_value()));
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
            return input.storage.at(access.storage).slot_count > 1;
          });
        });
      });
  if (input.region_tree) return AnalyzeNumericalSync(std::move(input), composition_order);
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
  // A child is contiguous. Crossings can only touch its first/last storage
  // accesses; queries through a child use its retained periodic event graph.
  for (size_t r = 0; r < result.segments.size(); ++r) {
    const auto& s = result.segments[r];
    if (!s.count) continue;
    for (int64_t i : std::set<int64_t>{0, s.count - 1})
      for (size_t p = 0; p < s.pipes.size(); ++p)
        for (bool c : {false, true}) result.Node({{r, i, p}, c});
  }
  std::vector<std::vector<std::pair<CompactSyncResult::Event, size_t>>> ports(result.segments.size());
  for (const auto& [event, index] : result.nodes_) ports[std::get<0>(event.first)].emplace_back(event, index);
  for (const auto& child : ports)
    for (const auto& [a, ai] : child)
      for (const auto& [b, bi] : child) {
        ++result.port_pair_tests;
        auto [ar, av, as] = a.first;
        auto [br, bv, bs] = b.first;
        if (result.segments[ar].Precedes(av, as, a.second, bv, bs, b.second)) result.graph_[ai].push_back(bi);
      }
  std::array<std::optional<Occurrence>, kPipeCount> previous{};
  std::vector<std::optional<Occurrence>> writers(atoms.size());
  std::vector<std::array<std::optional<Occurrence>, kPipeCount>> readers(atoms.size());
  std::set<Crossing> crossings;
  for (size_t r = 0; r < result.segments.size(); ++r) {
    const auto& s = result.segments[r];
    if (!s.count || s.pipes.empty()) continue;
    std::array<std::optional<size_t>, kPipeCount> first{}, last{};
    for (size_t p = 0; p < s.pipes.size(); ++p) {
      if (!first[s.pipes[p]]) first[s.pipes[p]] = p;
      last[s.pipes[p]] = p;
    }
    for (size_t p = 0; p < kPipeCount; ++p)
      if (first[p]) {
        if (previous[p])
          for (bool c : {false, true})
            result.graph_[result.Node({*previous[p], c})].push_back(result.Node({{r, 0, *first[p]}, c}));
        previous[p] = Occurrence{r, s.count - 1, *last[p]};
      }
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
  for (const auto& e : crossings)
    result.graph_[result.Node({e.first, true})].push_back(result.Node({e.second, false}));
  for (const auto& e : crossings) {
    ++result.crossing_path_searches;
    auto a = result.Node({e.first, true}), b = result.Node({e.second, false});
    if (!result.Path(a, b, std::make_pair(a, b))) result.crossing_demands.push_back(e);
  }
  for (const auto& g : result.graph_) result.graph_edges += g.size();
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
  std::vector<size_t> starts, ends;
  for (const auto& [event, node] : nodes_) {
    auto [r, v, s] = event.first;
    if (r == ar &&
        ((event.first == a && event.second == sc) || segments[ar].Precedes(av, as, sc, v, s, event.second)))
      starts.push_back(node);
    if (r == br &&
        ((event.first == b && event.second == tc) || segments[br].Precedes(v, s, event.second, bv, bs, tc)))
      ends.push_back(node);
  }
  for (auto x : starts)
    for (auto y : ends)
      if (Path(x, y)) return true;
  return false;
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
      CHECK(access.storage != storage || !access.coordinate_frame ||
            access.coordinate_frame == input.segments[region].original_loop)
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
bool CompactSyncResult::ScopedConsecutiveReuse(size_t family) const {
  if (auto found = scoped_reuse_.find(family); found != scoped_reuse_.end()) {
    return found->second;
  }
  const bool certified = CertifyNumericalConsecutiveReuse(*this, family);
  scoped_reuse_.emplace(family, certified);
  return certified;
}
bool CompactSyncResult::ScopedAdjacentBarrier(size_t family) const {
  return CertifyNumericalAdjacentBarrier(*this, family);
}
bool CompactSyncResult::ReferenceBefore(const Occurrence& a, const Occurrence& b) const {
  return hierarchy ? NumericalReferenceBefore(*this, a, b) : a < b;
}
std::vector<Crossing> CompactSyncResult::ExpandRetainedDemands(size_t max_payloads) const {
  return ExpandNumericalDemands(*this, max_payloads);
}
}  // namespace pypto::ir::sync
