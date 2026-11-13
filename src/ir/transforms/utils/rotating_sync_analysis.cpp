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

#include "src/ir/transforms/utils/rotating_sync_analysis.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto::ir::sync {
namespace {
using Wide = __int128;
int64_t Mod(Wide value, int64_t modulus) {
  const Wide remainder = value % modulus;
  return static_cast<int64_t>(remainder < 0 ? remainder + modulus : remainder);
}
Wide Add(Wide a, Wide b) {
  Wide result;
  CHECK(!__builtin_add_overflow(a, b, &result)) << "Periodic frontier arithmetic exceeds signed wide range";
  return result;
}
Wide Multiply(Wide a, Wide b) {
  Wide result;
  CHECK(!__builtin_mul_overflow(a, b, &result)) << "Periodic frontier arithmetic exceeds signed wide range";
  return result;
}
int64_t Inverse(int64_t a, int64_t modulus) {
  if (modulus == 1) return 0;
  Wide old_r = a, r = modulus, old_t = 1, t = 0;
  while (r) {
    const Wide q = old_r / r;
    const Wide next_r = old_r - q * r, next_t = old_t - q * t;
    old_r = r;
    r = next_r;
    old_t = t;
    t = next_t;
  }
  CHECK(old_r == 1) << "Rotating normalized stride has no inverse";
  return Mod(old_t, modulus);
}
struct Fragment {
  size_t site;
  int64_t offset;
  bool write;
};
struct Key {
  int64_t phase;
  size_t site;
  bool write;
};
using Generator = std::tuple<size_t, size_t, int64_t>;
using AccessRef = std::pair<size_t, const SyncAccess*>;

// A translated envelope is a sufficient physical separation certificate only
// in an admitted injective coordinate domain. Include reads as well as writes:
// a small write alone does not exclude cross-visit read/write conflicts.
std::optional<VisitOwnedStorage> VisitOwnership(const SyncStorage& storage,
                                                const std::vector<AccessRef>& accesses, int64_t count) {
  if (accesses.empty() || !std::any_of(accesses.begin(), accesses.end(),
                                       [](const AccessRef& access) { return access.second->write; }))
    return std::nullopt;
  const auto& first = *accesses.front().second;
  if (!std::any_of(accesses.begin(), accesses.end(), [](const AccessRef& access) {
        return access.second->row_step || access.second->column_step;
      }))
    return std::nullopt;
  CHECK(storage.slot_count == 1 && (storage.space == MemorySpace::DDR || storage.byte_coordinates))
      << "Visit-owned storage requires an injective ND GM or physical-byte domain";
  VisitOwnedStorage proof{first.row_step, first.column_step, first.row,
                          first.row,      first.column,      first.column};
  for (const auto& [site, pointer] : accesses) {
    const auto& access = *pointer;
    CHECK(access.row_step == proof.row_step && access.column_step == proof.column_step)
        << "Visit-owned storage requires common translation across every read and write";
    CHECK(access.row >= 0 && access.column >= 0 && access.rows > 0 && access.columns > 0 &&
          Wide(access.row) + access.rows <= storage.rows &&
          Wide(access.column) + access.columns <= storage.columns)
        << "Visit-owned access exceeds the supplied physical domain";
    if (count) {
      CHECK(Wide(access.row) + Wide(count - 1) * access.row_step + access.rows <= storage.rows &&
            Wide(access.column) + Wide(count - 1) * access.column_step + access.columns <= storage.columns)
          << "Visit-owned translation exceeds the supplied physical domain";
    }
    proof.row_begin = std::min(proof.row_begin, access.row);
    proof.column_begin = std::min(proof.column_begin, access.column);
    proof.row_end = std::max(proof.row_end, access.row + access.rows);
    proof.column_end = std::max(proof.column_end, access.column + access.columns);
  }
  CHECK((proof.row_step > 0 && Wide(proof.row_end) - proof.row_begin <= proof.row_step) ||
        (proof.column_step > 0 && Wide(proof.column_end) - proof.column_begin <= proof.column_step))
      << "Visit-owned storage lacks a disjoint all-access envelope";
  return proof;
}
void ExtractVisitOwned(const std::vector<Fragment>& fragments, CompactSegmentResult& result,
                       std::set<Generator>& generators) {
  // Normalization orders one-slot fragments by original body position and
  // coalesces RMW duplicates. No writer from another invocation is eligible.
  std::optional<size_t> previous, next;
  for (const auto& fragment : fragments) {
    ++result.sweep_visits;
    if (previous) generators.emplace(*previous, fragment.site, 0);
    if (fragment.write) previous = fragment.site;
  }
  for (auto i = fragments.rbegin(); i != fragments.rend(); ++i) {
    ++result.sweep_visits;
    if (!i->write && next) generators.emplace(i->site, *next, 0);
    if (i->write) next = i->site;
  }
}

// Geometry normalization is charged separately from the circular extraction.
// Full-slot families have just one atom. Static overlapping rectangles can
// produce more atoms; this cost is not called linear in source accesses.
std::vector<std::vector<Fragment>> Normalize(const SyncStorage& storage,
                                             const std::vector<AccessRef>& accesses,
                                             CompactSegmentResult& result, bool visit_owned) {
  std::vector<std::vector<Fragment>> atoms;
  const bool writes = std::any_of(accesses.begin(), accesses.end(),
                                  [](const AccessRef& entry) { return entry.second->write; });
  if (!writes) return atoms;
  std::set<int64_t> row_cuts, column_cuts;
  for (const auto& [site, access] : accesses) {
    CHECK(visit_owned || (access->row_step == 0 && access->column_step == 0))
        << "Moving accesses sharing a written domain need a separate exact generator certificate";
    CHECK(access->rows > 0 && access->columns > 0 && access->row >= 0 && access->column >= 0 &&
          Wide(access->row) + access->rows <= storage.rows &&
          Wide(access->column) + access->columns <= storage.columns)
        << "Rotating access geometry lies outside its supplied storage domain";
    row_cuts.insert(access->row);
    row_cuts.insert(access->row + access->rows);
    column_cuts.insert(access->column);
    column_cuts.insert(access->column + access->columns);
  }
  std::vector<int64_t> rows(row_cuts.begin(), row_cuts.end()),
      columns(column_cuts.begin(), column_cuts.end());
  for (size_t r = 1; r < rows.size(); ++r) {
    for (size_t c = 1; c < columns.size(); ++c) {
      std::map<std::pair<size_t, int64_t>, bool> fragments;
      for (const auto& [site, access] : accesses) {
        ++result.atom_tests;
        if (access->row > rows[r - 1] || access->row + access->rows < rows[r] ||
            access->column > columns[c - 1] || access->column + access->columns < columns[c])
          continue;
        // Equal read/write fragments at one payload are one RMW writer key.
        fragments[{site, Mod(access->slot_first, storage.slot_count)}] |= access->write;
      }
      if (fragments.empty()) continue;
      auto& atom = atoms.emplace_back();
      for (const auto& [key, write] : fragments) atom.push_back({key.first, key.second, write});
      result.rotating_fragments += atom.size();
    }
  }
  return atoms;
}

void Extract(const std::vector<Fragment>& fragments, const RotatingSlotInfo& slot,
             CompactSegmentResult& result, std::set<Generator>& generators) {
  std::map<int64_t, std::vector<Key>> groups;
  for (const auto& fragment : fragments) {
    const int64_t residue = fragment.offset % slot.gcd;
    const int64_t quotient = fragment.offset / slot.gcd;
    groups[residue].push_back(
        {Mod(-Wide(slot.inverse) * quotient, slot.period), fragment.site, fragment.write});
  }
  for (auto& [residue, keys] : groups) {
    ++result.modular_groups;
    std::sort(keys.begin(), keys.end(), [&](const Key& a, const Key& b) {
      ++result.sort_comparisons;
      return std::tie(a.phase, a.site) < std::tie(b.phase, b.site);
    });
    std::optional<size_t> first, last;
    for (size_t i = 0; i < keys.size(); ++i) {
      ++result.sweep_visits;
      if (keys[i].write) {
        if (!first) first = i;
        last = i;
      }
    }
    if (!first) continue;
    size_t previous = *last;
    for (size_t i = 0; i < keys.size(); ++i) {
      ++result.sweep_visits;
      int64_t distance = Mod(Wide(keys[i].phase) - keys[previous].phase, slot.period);
      if (distance == 0 && keys[previous].site >= keys[i].site) distance = slot.period;
      generators.emplace(keys[previous].site, keys[i].site, distance);
      if (keys[i].write) previous = i;
    }
    size_t next = *first;
    for (size_t i = keys.size(); i-- > 0;) {
      ++result.sweep_visits;
      if (!keys[i].write) {
        int64_t distance = Mod(Wide(keys[next].phase) - keys[i].phase, slot.period);
        if (distance == 0 && keys[next].site <= keys[i].site) distance = slot.period;
        generators.emplace(keys[i].site, keys[next].site, distance);
      }
      if (keys[i].write) next = i;
    }
  }
}
// Range assignment/maximum over compressed endpoints, never over bytes or
// interval/atom incidences. Zero is the absent key; positive keys increase in
// circular sweep order. A pending assignment covers this entire tree node.
class IntervalPredecessors {
 public:
  IntervalPredecessors(size_t size, CompactSegmentResult& work) : size_(size), work_(work), nodes_(4 * size) {
    work_.interval_tree_entries += nodes_.size();
  }
  void Assign(size_t begin, size_t end, size_t key) {
    ++work_.interval_updates;
    Assign(1, 0, size_, begin, end, key);
  }
  size_t Maximum(size_t begin, size_t end) {
    ++work_.interval_queries;
    return Maximum(1, 0, size_, begin, end);
  }

 private:
  struct Node {
    size_t maximum = 0;
    std::optional<size_t> assignment;
  };
  void Set(size_t node, size_t key) { nodes_[node] = {key, key}; }
  void Push(size_t node) {
    if (!nodes_[node].assignment) return;
    Set(2 * node, *nodes_[node].assignment);
    Set(2 * node + 1, *nodes_[node].assignment);
    nodes_[node].assignment.reset();
  }
  void Assign(size_t node, size_t lo, size_t hi, size_t begin, size_t end, size_t key) {
    ++work_.interval_node_visits;
    if (begin <= lo && hi <= end) {
      Set(node, key);
      return;
    }
    Push(node);
    const size_t mid = lo + (hi - lo) / 2;
    if (begin < mid) Assign(2 * node, lo, mid, begin, end, key);
    if (end > mid) Assign(2 * node + 1, mid, hi, begin, end, key);
    nodes_[node].maximum = std::max(nodes_[2 * node].maximum, nodes_[2 * node + 1].maximum);
  }
  size_t Maximum(size_t node, size_t lo, size_t hi, size_t begin, size_t end) {
    ++work_.interval_node_visits;
    if (begin <= lo && hi <= end) return nodes_[node].maximum;
    Push(node);
    const size_t mid = lo + (hi - lo) / 2;
    size_t answer = 0;
    if (begin < mid) answer = Maximum(2 * node, lo, mid, begin, end);
    if (end > mid) answer = std::max(answer, Maximum(2 * node + 1, mid, hi, begin, end));
    return answer;
  }
  size_t size_;
  CompactSegmentResult& work_;
  std::vector<Node> nodes_;
};
struct IntervalKey {
  int64_t phase;
  size_t site;
  bool write;
  int64_t begin, end;
  size_t lo = 0, hi = 0;
};
// Target/source-pipe -> (distance, source). Compare reconstructed source
// identities (-distance, site), never phases from different families.
using IntervalCandidates = std::map<std::pair<size_t, PipeType>, std::pair<int64_t, size_t>>;
void ExtractIntervals(const SyncStorage& storage, const std::vector<AccessRef>& accesses,
                      const RotatingSlotInfo& slot, CompactSegmentResult& result,
                      IntervalCandidates& candidates) {
  ++result.interval_domains;
  std::map<int64_t, std::vector<IntervalKey>> groups;
  for (const auto& [site, access] : accesses) {
    const int64_t offset = Mod(access->slot_first, storage.slot_count);
    const auto [begin, end] = *access->physical_interval;
    INTERNAL_CHECK(begin >= 0 && begin < end) << "Invalid retained exact physical interval";
    groups[offset % slot.gcd].push_back(
        {Mod(-Wide(slot.inverse) * (offset / slot.gcd), slot.period), site, access->write, begin, end});
    ++result.interval_fragments;
  }
  for (auto& [residue, keys] : groups) {
    ++result.modular_groups;
    std::vector<int64_t> endpoints;
    std::set<PipeType> pipes;
    for (const auto& key : keys) {
      endpoints.push_back(key.begin);
      endpoints.push_back(key.end);
      pipes.insert(result.pipes[key.site]);
    }
    std::sort(endpoints.begin(), endpoints.end(), [&](int64_t a, int64_t b) {
      ++result.sort_comparisons;
      return a < b;
    });
    endpoints.erase(std::unique(endpoints.begin(), endpoints.end()), endpoints.end());
    result.interval_endpoints += endpoints.size();
    for (auto& key : keys) {
      key.lo = std::lower_bound(endpoints.begin(), endpoints.end(), key.begin) - endpoints.begin();
      key.hi = std::lower_bound(endpoints.begin(), endpoints.end(), key.end) - endpoints.begin();
    }
    std::sort(keys.begin(), keys.end(), [&](const IntervalKey& a, const IntervalKey& b) {
      ++result.sort_comparisons;
      return std::tie(a.phase, a.site) < std::tie(b.phase, b.site);
    });
    std::map<PipeType, std::pair<IntervalPredecessors, IntervalPredecessors>> trees;
    for (auto pipe : pipes)
      trees.emplace(pipe, std::make_pair(IntervalPredecessors(endpoints.size() - 1, result),
                                         IntervalPredecessors(endpoints.size() - 1, result)));
    // The first copy supplies all wraparound predecessors. Equal (phase,site)
    // fragments form one occurrence: query every mode/view before any update.
    // Different offsets at one site may denote different canonical visits.
    for (size_t copy = 0; copy < 2; ++copy) {
      for (size_t first = 0; first < keys.size();) {
        size_t last = first + 1;
        while (last < keys.size() && keys[last].phase == keys[first].phase &&
               keys[last].site == keys[first].site)
          ++last;
        for (size_t index = first; index < last; ++index) {
          const auto& target = keys[index];
          ++result.sweep_visits;
          for (auto& [pipe, pair] : trees) {
            const size_t answer = (target.write ? pair.second : pair.first).Maximum(target.lo, target.hi);
            if (!copy || !answer) continue;
            const size_t ordinal = answer - 1;
            const auto& source = keys[ordinal % keys.size()];
            const Wide distance =
                Wide(target.phase) + slot.period - source.phase - Wide(ordinal / keys.size()) * slot.period;
            INTERNAL_CHECK(distance >= 0 && distance <= slot.period)
                << "Interval predecessor is outside the preceding orbit";
            const auto candidate = std::make_pair(static_cast<int64_t>(distance), source.site);
            const auto name = std::make_pair(target.site, pipe);
            auto found = candidates.find(name);
            if (found == candidates.end() || candidate.first < found->second.first ||
                (candidate.first == found->second.first && candidate.second > found->second.second))
              candidates[name] = candidate;
          }
        }
        for (size_t index = first; index < last; ++index) {
          const auto& key = keys[index];
          auto& pair = trees.at(result.pipes[key.site]);
          const size_t ordinal = 1 + copy * keys.size() + index;
          pair.second.Assign(key.lo, key.hi, ordinal);
          if (key.write) pair.first.Assign(key.lo, key.hi, ordinal);
        }
        first = last;
      }
    }
  }
}
struct Edge {
  size_t source, target;
  int64_t weight;
};
struct BestTwo {
  struct Entry {
    Wide value;
    size_t edge;
  };
  std::optional<Entry> first, second;
  void Insert(Wide value, size_t edge) {
    if (!first || value > first->value) {
      second = first;
      first = Entry{value, edge};
    } else if (!second || value > second->value) {
      second = Entry{value, edge};
    }
  }
  std::optional<Wide> Excluding(size_t edge) const {
    const auto& entry = first && first->edge != edge ? first : second;
    return entry ? std::optional<Wide>(entry->value) : std::nullopt;
  }
};
size_t Reduce(CompactSegmentResult& result, bool extract_covers = true) {
  const size_t sites = result.pipes.size(), vertices = 2 * sites;
  std::array<std::vector<size_t>, kPipeCount> pipes;
  result.ranks.resize(sites);
  for (size_t site = 0; site < sites; ++site) {
    auto& pipe = pipes.at(static_cast<size_t>(result.pipes[site]));
    pipe.push_back(site);
    result.ranks[site] = static_cast<int64_t>(pipe.size());
  }
  std::vector<Edge> edges;
  std::vector<std::vector<size_t>> outgoing(vertices);
  auto edge = [&](size_t source, size_t target, int64_t weight) {
    outgoing[source].push_back(edges.size());
    edges.push_back({source, target, weight});
  };
  for (size_t site = 0; site < sites; ++site) edge(2 * site, 2 * site + 1, 0);
  for (size_t p = 0; p < kPipeCount; ++p) {
    const auto& pipe = pipes[p];
    result.pipe_sizes[p] = static_cast<int64_t>(pipe.size());
    for (size_t i = 0; i < pipe.size(); ++i) {
      const size_t a = pipe[i], b = pipe[(i + 1) % pipe.size()];
      const int64_t wrap = i + 1 == pipe.size() ? 1 : 0;
      edge(2 * a, 2 * b, wrap);
      edge(2 * a + 1, 2 * b + 1, wrap);
    }
  }
  const size_t generator_begin = edges.size();
  for (const auto& generator : result.generators)
    edge(2 * generator.source + 1, 2 * generator.target, generator.distance);
  std::vector<bool> redundant(result.generators.size());
  for (size_t p = 0; p < kPipeCount; ++p) {
    const int64_t h = result.pipe_sizes[p];
    if (!h) continue;
    for (size_t completion = 0; completion < 2; ++completion) {
      ++result.shortest_path_runs;
      auto& distances = result.frontier_distances[p + completion * kPipeCount];
      distances.resize(vertices);
      result.retained_frontier_entries += vertices;
      using Work = std::pair<Wide, size_t>;
      std::priority_queue<Work, std::vector<Work>, std::greater<Work>> queue;
      for (size_t site : pipes[p]) {
        const size_t event = 2 * site + completion;
        distances[event] = h - result.ranks[site];
        queue.emplace(*distances[event], event);
      }
      while (!queue.empty()) {
        const auto [distance, node] = queue.top();
        queue.pop();
        if (distance != distances[node]) continue;
        for (size_t id : outgoing[node]) {
          ++result.shortest_path_edge_visits;
          const auto& e = edges[id];
          const Wide next = Add(distance, Multiply(h, e.weight));
          if (!distances[e.target] || next < *distances[e.target]) {
            distances[e.target] = next;
            queue.emplace(next, e.target);
          }
        }
      }
      if (!completion || !extract_covers) continue;
      std::vector<BestTwo> best(vertices);
      for (size_t id = 0; id < edges.size(); ++id) {
        const auto& e = edges[id];
        if (distances[e.source])
          best[e.target].Insert(Add(Wide(h) - *distances[e.source], -Multiply(h, e.weight)), id);
      }
      for (size_t i = 0; i < result.generators.size(); ++i) {
        const auto& g = result.generators[i];
        if (static_cast<size_t>(result.pipes[g.source]) != p) continue;
        const auto alternative = best[2 * g.target].Excluding(generator_begin + i);
        redundant[i] = alternative && Add(Multiply(h, g.distance), *alternative) >= result.ranks[g.source];
      }
    }
  }
  if (extract_covers) {
    for (size_t i = 0; i < result.generators.size(); ++i)
      if (!redundant[i]) result.demands.push_back(result.generators[i]);
  }
  return edges.size();
}
Wide Floor(Wide numerator, int64_t denominator) {
  const Wide q = numerator / denominator, r = numerator % denominator;
  return q - (r < 0);
}
Wide Ceil(Wide numerator, int64_t denominator) { return -Floor(-numerator, denominator); }
std::optional<std::pair<int64_t, int64_t>> PointVisits(const SyncAccess& a, int64_t count, int64_t row,
                                                       int64_t column) {
  if (!count) return std::nullopt;
  Wide low = 0, high = count - 1;
  auto axis = [&](int64_t x, int64_t origin, int64_t step, int64_t extent) {
    if (!step) {
      if (Wide(x) < origin || Wide(x) >= Wide(origin) + extent) high = -1;
    } else {
      low = std::max(low, Ceil(Wide(x) - origin - extent + 1, step));
      high = std::min(high, Floor(Wide(x) - origin, step));
    }
  };
  axis(row, a.row, a.row_step, a.rows);
  axis(column, a.column, a.column_step, a.columns);
  if (low > high) return std::nullopt;
  return std::make_pair(static_cast<int64_t>(low), static_cast<int64_t>(high));
}
std::optional<std::pair<int64_t, int64_t>> Progression(int64_t low, int64_t high, int64_t phase,
                                                       int64_t period) {
  if (low > high) return std::nullopt;
  const Wide first = Wide(low) + Mod(Wide(phase) - low, period);
  const Wide last = Wide(high) - Mod(Wide(high) - phase, period);
  if (first > last) return std::nullopt;
  return std::make_pair(static_cast<int64_t>(first), static_cast<int64_t>(last));
}
}  // namespace
size_t BuildPeriodicCoreFrontiers(CompactSegmentResult& result) { return Reduce(result, false); }

CompactSyncResult AnalyzeRotatingSync(CompactSyncInput input) {
  CHECK(input.segments.size() == 1)
      << "Rotating child mathematics is supported; multi-region composition export is unavailable";
  CompactSyncResult result;
  result.input = std::move(input);
  result.segments.push_back(AnalyzeRotatingSegment(result.input, 0, result.input.segments[0].count, result));
  return result;
}
CompactSegmentResult AnalyzeRotatingSegment(const CompactSyncInput& input, size_t region, int64_t count,
                                            CompactSyncResult& work) {
  CompactSegmentResult output;
  const auto& segment = input.segments.at(region);
  CHECK(count >= 0) << "Negative rotating invocation count";
  CHECK(segment.sites.size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max() / 2))
      << "Rotating body exceeds supported event index range";
  output.rotating = true;
  output.count = count;
  output.internal_scope_count = count;
  std::map<size_t, std::vector<AccessRef>> accesses;
  std::map<size_t, std::optional<int64_t>> stride;
  for (size_t s = 0; s < segment.sites.size(); ++s) {
    const auto& site = segment.sites[s];
    CHECK(!site.continuation && !site.protected_accumulator && !site.initializes_accumulator)
        << "Rotating analysis needs invariant payload effects; accumulator/continuation export unavailable";
    CHECK(static_cast<size_t>(site.pipe) < kPipeCount) << "Invalid rotating payload pipe";
    output.pipes.push_back(site.pipe);
    for (size_t i = 0; i < site.accesses.size(); ++i) {
      const auto& a = site.accesses[i];
      CHECK(a.storage < input.storage.size() && !a.after_first)
          << "Rotating access requires a covered domain and unconditional body membership";
      const auto& storage = input.storage[a.storage];
      CHECK(storage.slot_count > 0 && storage.rows > 0 && storage.columns > 0 && a.row_step >= 0 &&
            a.column_step >= 0 && a.rows > 0 && a.columns > 0)
          << "Unsupported rotating storage geometry or affine direction";
      const int64_t step = Mod(a.slot_step, storage.slot_count);
      CHECK(!stride[a.storage] || *stride[a.storage] == step)
          << "Rotating family requires one common normalized stride";
      stride[a.storage] = step;
      accesses[a.storage].emplace_back(s, &a);
      output.storage_accesses[a.storage].emplace_back(s, i);
    }
  }
  using Clock = std::chrono::steady_clock;
  const auto start = Clock::now();
  std::set<Generator> generators;
  IntervalCandidates interval_candidates;
  for (const auto& [storage, fragments] : accesses) {
    const auto& domain = input.storage[storage];
    CHECK(domain.slot_count > 0) << "Invalid rotating slot count";
    auto& slot = output.slot_info[storage];
    slot.stride = stride[storage].value_or(0);
    slot.gcd = std::gcd(slot.stride, domain.slot_count);
    slot.period = domain.slot_count / slot.gcd;
    slot.inverse = Inverse(slot.stride / slot.gcd, slot.period);
    const auto ownership = VisitOwnership(domain, fragments, count);
    if (ownership) output.visit_owned.emplace(storage, *ownership);
    // A partial or unqualified domain stays on its established exact modeled
    // geometry path. Never omit its accesses or fabricate interval evidence.
    if (!ownership && std::all_of(fragments.begin(), fragments.end(), [](const AccessRef& access) {
          return access.second->physical_interval.has_value();
        })) {
      ExtractIntervals(domain, fragments, slot, output, interval_candidates);
      continue;
    }
    auto atoms = Normalize(domain, fragments, output, ownership.has_value());
    work.cells += atoms.size();
    for (const auto& atom : atoms) {
      if (ownership)
        ExtractVisitOwned(atom, output, generators);
      else
        Extract(atom, slot, output, generators);
    }
  }
  for (const auto& [target_pipe, source] : interval_candidates)
    generators.emplace(source.second, target_pipe.first, source.first);
  for (const auto& [source, target, distance] : generators) {
    CHECK(distance >= 0 && (distance > 0 || source < target))
        << "Rotating generator is not reference-forward";
    output.generators.push_back({source, target, distance});
  }
  const auto extracted = Clock::now();
  work.partition_seconds += std::chrono::duration<double>(extracted - start).count();
  output.quotient_edges = Reduce(output);
  work.reduction_seconds += std::chrono::duration<double>(Clock::now() - extracted).count();
  // This is the static number of described sites, not an unrolled witness.
  work.analyzed_occurrences += segment.sites.size();
  return output;
}

PointBoundary RotatingBoundary(const CompactSyncInput& input, const CompactSegmentResult& segment,
                               size_t storage, int64_t slot, int64_t row, int64_t column, size_t region) {
  PointBoundary result;
  const auto& input_segment = input.segments.at(region);
  const auto accesses = segment.storage_accesses.find(storage);
  if (accesses == segment.storage_accesses.end()) return result;
  const auto& info = segment.slot_info.at(storage);
  struct Range {
    size_t site;
    PipeType pipe;
    int64_t low, high, phase;
    bool write;
  };
  std::vector<Range> ranges;
  std::unordered_map<size_t, std::unordered_set<int64_t>> writer_phases;
  for (const auto& [site_index, access_index] : accesses->second) {
    const auto& site = input_segment.sites[site_index];
    const auto& access = site.accesses[access_index];
    const Wide difference = Wide(slot) - access.slot_first;
    if (difference % info.gcd != 0) continue;
    const auto visits = PointVisits(access, segment.count, row, column);
    if (!visits) continue;
    const int64_t phase = Mod(Wide(info.inverse) * (difference / info.gcd), info.period);
    const auto selected = Progression(visits->first, visits->second, phase, info.period);
    if (!selected) continue;
    ranges.push_back({site_index, site.pipe, selected->first, selected->second, phase, access.write});
    if (access.write) {
      writer_phases[site_index].insert(phase);
      const Occurrence first{0, selected->first, site_index}, last{0, selected->second, site_index};
      if (!result.first_writer || first < *result.first_writer) result.first_writer = first;
      if (!result.last_writer || last > *result.last_writer) result.last_writer = last;
    }
  }
  for (const auto& range : ranges) {
    if (range.write) continue;
    const auto writers = writer_phases.find(range.site);
    if (writers != writer_phases.end() && writers->second.count(range.phase)) continue;
    int64_t prefix_end = range.high, suffix_start = range.low;
    if (result.first_writer) {
      const auto [region, visit, site] = *result.first_writer;
      prefix_end = std::min(prefix_end, visit - (range.site >= site));
    }
    if (result.last_writer) {
      const auto [region, visit, site] = *result.last_writer;
      suffix_start = std::max(suffix_start, visit + (range.site <= site));
    }
    if (const auto prefix = Progression(range.low, prefix_end, range.phase, info.period)) {
      const Occurrence first{0, prefix->first, range.site};
      if (!result.first_readers[range.pipe] || first < *result.first_readers[range.pipe])
        result.first_readers[range.pipe] = first;
    }
    if (const auto suffix = Progression(suffix_start, range.high, range.phase, info.period)) {
      const Occurrence last{0, suffix->second, range.site};
      if (!result.last_readers[range.pipe] || last > *result.last_readers[range.pipe])
        result.last_readers[range.pipe] = last;
    }
  }
  return result;
}
}  // namespace pypto::ir::sync
