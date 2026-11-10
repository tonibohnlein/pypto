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

#include "numerical_sync_hierarchy.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <limits>
#include <numeric>
#include <set>
#include <unordered_map>

#include "numerical_sync_merge.h"
#include "pypto/core/logging.h"
#include "rotating_sync_analysis.h"
namespace pypto::ir::sync {
namespace {
using Ordinal = NumericalOrdinal;
using Node = NumericalNode;
using NodePtr = std::shared_ptr<const Node>;
using Row = NumericalRow;
Ordinal Length(int64_t count, size_t sites) { return Ordinal(count) * sites; }
int64_t Scope(const SyncSegment& segment) {
  return segment.nested_frames.empty() ? segment.count : segment.nested_frames.back().count;
}
int64_t Repetitions(const SyncSegment& segment) {
  Ordinal repetitions = 1;
  for (size_t i = 0; i + 1 < segment.nested_frames.size(); ++i) {
    repetitions *= segment.nested_frames[i].count;
    CHECK(repetitions <= std::numeric_limits<int64_t>::max()) << "Numerical repetition count overflow";
  }
  return static_cast<int64_t>(repetitions);
}
int64_t Mod(Ordinal value, int64_t modulus) {
  const auto residue = value % modulus;
  return static_cast<int64_t>(residue < 0 ? residue + modulus : residue);
}
NumericalBoundary LeafBoundary(const CompactSyncResult& result, size_t leaf, const NumericalCell& cell) {
  const auto selected = RotatingBoundary(result.input, result.segments[leaf], cell.storage, cell.slot,
                                         cell.row, cell.column, leaf);
  const size_t sites = result.input.segments[leaf].sites.size();
  auto ordinal = [&](const std::optional<Occurrence>& occurrence) -> std::optional<Ordinal> {
    if (!occurrence) return std::nullopt;
    return Length(std::get<1>(*occurrence), sites) + std::get<2>(*occurrence);
  };
  NumericalBoundary out;
  out.first_writer = ordinal(selected.first_writer);
  out.last_writer = ordinal(selected.last_writer);
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = ordinal(selected.first_readers[p]);
    out.last_readers[p] = ordinal(selected.last_readers[p]);
  }
  return out;
}
std::vector<NumericalCell> InterfaceCells(CompactSyncResult& result) {
  const auto& input = result.input;
  std::vector<bool> written(input.storage.size());
  using Effect = std::pair<const SyncSegment*, const SyncAccess*>;
  std::map<size_t, std::vector<Effect>> effects;
  for (const auto& segment : input.segments)
    for (const auto& site : segment.sites)
      for (const auto& access : site.accesses) {
        ++result.interface_access_visits;
        if (access.write) written.at(access.storage) = true;
        effects[access.storage].emplace_back(&segment, &access);
      }
  std::map<size_t, size_t> private_owner;
  for (size_t region = 0; region < result.segments.size(); ++region)
    for (const auto& [storage, certificate] : result.segments[region].visit_owned) {
      const auto [owner, inserted] = private_owner.emplace(storage, region);
      CHECK(inserted || owner->second == region)
          << "Visit-owned storage has multiple region owners; cross-reset selector export unavailable";
    }
  std::vector<NumericalCell> cells;
  for (const auto& [storage, domain_effects] : effects) {
    if (const auto owner = private_owner.find(storage); owner != private_owner.end()) {
      const auto& region = input.segments[owner->second];
      CHECK(region.nested_frames.empty())
          << "Visit-owned storage cannot be omitted across repeated or reset invocations";
      for (const auto& [segment, access] : domain_effects) {
        ++result.interface_access_visits;
        CHECK(segment == &region)
            << "Visit-owned storage has an outside read or write; crossing selector export unavailable";
      }
      ++result.omitted_bridge_domains;
      continue;
    }
    // The shared alias-qualified physical inventory proves that no region
    // writes this domain. Its point selectors remain available on demand.
    if (!written[storage]) continue;
    std::set<int64_t> rows, columns, slots;
    for (const auto& [segment, pointer] : domain_effects) {
      ++result.interface_access_visits;
      const auto& access = *pointer;
      if (!segment->count) continue;
      const auto count = Scope(*segment);
      CHECK(!access.row_step && !access.column_step)
          << "Moving geometry sharing a written regional domain lacks a complete selector partition";
      rows.insert(access.row);
      rows.insert(access.row + access.rows);
      columns.insert(access.column);
      columns.insert(access.column + access.columns);
      const int64_t banks = input.storage[storage].slot_count;
      const int64_t period = banks / std::gcd(access.slot_step, banks);
      // Explicit complete-interface output work. Enumerate visited physical
      // slots only; never the trip count beyond one orbit or a joint period.
      for (int64_t visit = 0; visit < std::min(count, period); ++visit) {
        ++result.interface_slot_visits;
        slots.insert(Mod(Ordinal(access.slot_first) + Ordinal(visit) * access.slot_step, banks));
      }
    }
    std::vector<int64_t> r(rows.begin(), rows.end()), c(columns.begin(), columns.end());
    for (int64_t slot : slots)
      for (size_t i = 1; i < r.size(); ++i)
        for (size_t j = 1; j < c.size(); ++j) cells.push_back({storage, slot, r[i - 1], c[j - 1]});
  }
  result.interface_cells = cells.size();
  return cells;
}
bool LeafQuery(const CompactSyncResult& result, size_t leaf, Ordinal a, bool sc, Ordinal b, bool tc) {
  if (a == b && sc == tc) return true;
  const auto sites = result.input.segments[leaf].sites.size();
  INTERNAL_CHECK(sites) << "Nonempty numerical leaf has no sites";
  return result.segments[leaf].Precedes(static_cast<int64_t>(a / sites), static_cast<size_t>(a % sites), sc,
                                        static_cast<int64_t>(b / sites), static_cast<size_t>(b % sites), tc);
}
NodePtr BuildLeaf(CompactSyncResult& result, const std::vector<NumericalCell>& cells, size_t leaf) {
  auto node = std::make_shared<Node>();
  node->leaf = leaf;
  const auto& input = result.input.segments[leaf];
  const auto count = result.segments[leaf].internal_scope_count;
  node->length = Length(count, input.sites.size());
  for (const auto& cell : cells) node->storage.push_back(LeafBoundary(result, leaf, cell));
  if (count) {
    for (size_t s = 0; s < input.sites.size(); ++s) {
      const auto p = input.sites[s].pipe;
      if (!node->first[p]) node->first[p] = s;
      node->last[p] = Length(count - 1, input.sites.size()) + s;
    }
  }
  for (const auto& event : NumericalPorts(*node, result.port_normalization_visits)) {
    const auto site = static_cast<size_t>(event.first % input.sites.size());
    const size_t chain = 2 * static_cast<size_t>(input.sites[site].pipe) + event.second;
    node->chains[chain].push_back(event.first);
  }
  IndexNumericalPositions(*node);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    node->forward[c].resize(node->chains[c].size());
    for (size_t d = 0; d < kNumericalChains; ++d) {
      size_t target = 0;
      for (size_t source = 0; source < node->chains[c].size(); ++source) {
        while (target < node->chains[d].size()) {
          ++result.interface_queries;
          if (LeafQuery(result, leaf, node->chains[c][source], c % 2, node->chains[d][target], d % 2)) break;
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
NodePtr Repeat(NodePtr body, int64_t count, CompactSyncResult& result) {
  NodePtr output = std::make_shared<Node>();
  if (!count || !body->length) return output;
  while (count) {
    if (count & 1) output = MergeNumericalNodes(output, body, result);
    count /= 2;
    if (count) body = MergeNumericalNodes(body, body, result);
  }
  return output;
}
NodePtr Compose(const std::vector<NodePtr>& nodes, size_t begin, size_t end, const std::string& order,
                CompactSyncResult& result) {
  if (begin == end) return std::make_shared<Node>();
  if (end - begin == 1) return nodes[begin];
  if (order == "left") {
    auto node = nodes[begin];
    for (size_t i = begin + 1; i < end; ++i) node = MergeNumericalNodes(node, nodes[i], result);
    return node;
  }
  if (order == "right") {
    auto node = nodes[end - 1];
    for (size_t i = end - 1; i-- > begin;) node = MergeNumericalNodes(nodes[i], node, result);
    return node;
  }
  const size_t middle = begin + (end - begin) / 2;
  return MergeNumericalNodes(Compose(nodes, begin, middle, order, result),
                             Compose(nodes, middle, end, order, result), result);
}
Occurrence Decode(const CompactSyncResult& result, Ordinal ordinal) {
  const auto& offsets = result.hierarchy->segment_offsets;
  const auto upper = std::upper_bound(offsets.begin(), offsets.end(), ordinal);
  CHECK(upper != offsets.begin() && upper != offsets.end()) << "Invalid numerical occurrence ordinal";
  const size_t region = static_cast<size_t>(upper - offsets.begin() - 1);
  const auto& segment = result.input.segments[region];
  const auto relative = ordinal - offsets[region];
  INTERNAL_CHECK(!segment.sites.empty()) << "Nonempty numerical region has no sites";
  return {region, static_cast<int64_t>(relative / segment.sites.size()),
          static_cast<size_t>(relative % segment.sites.size())};
}
Ordinal Encode(const CompactSyncResult& result, const Occurrence& occurrence) {
  const auto [region, visit, site] = occurrence;
  CHECK(region < result.input.segments.size()) << "Invalid numerical region";
  const auto& segment = result.input.segments[region];
  CHECK(visit >= 0 && visit < segment.count && site < segment.sites.size())
      << "Invalid numerical synchronization occurrence";
  return NumericalAdd(result.hierarchy->segment_offsets[region], Length(visit, segment.sites.size()) + site);
}
void ExpandNodeDemands(const CompactSyncResult& result, const NodePtr& node, Ordinal offset,
                       std::vector<Crossing>& out) {
  if (!node->length) return;
  if (!node->left) {
    const auto& segment = result.segments[node->leaf];
    const auto sites = result.input.segments[node->leaf].sites.size();
    for (const auto& demand : segment.demands)
      for (int64_t i = 0; i < segment.internal_scope_count - demand.distance; ++i) {
        const auto a = NumericalAdd(offset, Length(i, sites) + demand.source);
        const auto b = NumericalAdd(offset, Length(i + demand.distance, sites) + demand.target);
        out.emplace_back(Decode(result, a), Decode(result, b));
      }
    return;
  }
  ExpandNodeDemands(result, node->left, offset, out);
  const auto right_offset = NumericalAdd(offset, node->left->length);
  ExpandNodeDemands(result, node->right, right_offset, out);
  for (const auto& crossing : node->crossings) {
    if (crossing.native) continue;
    const auto a = NumericalAdd(offset, node->left->chains[crossing.source.first][crossing.source.second]);
    const auto b =
        NumericalAdd(right_offset, node->right->chains[crossing.target.first][crossing.target.second]);
    out.emplace_back(Decode(result, a), Decode(result, b));
  }
}
// Export only retained merge records: leaf demand families remain compact.
void ExportNodeCrossings(const CompactSyncResult& result, const NodePtr& node, Ordinal offset,
                         std::vector<Crossing>& out) {
  if (!node->length || !node->left) return;
  ExportNodeCrossings(result, node->left, offset, out);
  const auto right_offset = NumericalAdd(offset, node->left->length);
  ExportNodeCrossings(result, node->right, right_offset, out);
  for (const auto& crossing : node->crossings) {
    if (crossing.native) continue;
    const auto a = NumericalAdd(offset, node->left->chains[crossing.source.first][crossing.source.second]);
    const auto b =
        NumericalAdd(right_offset, node->right->chains[crossing.target.first][crossing.target.second]);
    out.emplace_back(Decode(result, a), Decode(result, b));
  }
}
Row EmptyForward(const Node& node) {
  Row out;
  for (size_t c = 0; c < kNumericalChains; ++c) out[c] = node.chains[c].size();
  return out;
}
Row EmptyReverse() {
  Row out;
  out.fill(-1);
  return out;
}
Row AcrossForward(const Node& node, const Row& source, const CompactSyncResult& result) {
  auto out = EmptyForward(*node.right);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t d = 0; d < kNumericalChains; ++d) {
      ++result.hierarchy_query_steps;
      const auto& list = node.crossing_index[c][d].forward;
      const auto found =
          std::lower_bound(list.begin(), list.end(), source[c],
                           [](const auto& entry, int64_t position) { return entry.first < position; });
      if (found != list.end()) out[d] = std::min(out[d], found->second);
    }
  }
  return out;
}
Row AcrossReverse(const Node& node, const Row& target, const CompactSyncResult& result) {
  auto out = EmptyReverse();
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t d = 0; d < kNumericalChains; ++d) {
      ++result.hierarchy_query_steps;
      const auto& list = node.crossing_index[c][d].reverse;
      const auto after =
          std::upper_bound(list.begin(), list.end(), target[d],
                           [](int64_t position, const auto& entry) { return position < entry.first; });
      if (after != list.begin()) out[c] = std::max(out[c], std::prev(after)->second);
    }
  }
  return out;
}
Row ProjectForward(const Node& node, const Row& left, const Row& right) {
  auto out = EmptyForward(node);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    const auto& l = node.left_projection[c];
    const auto& r = node.right_projection[c];
    const auto lp = std::lower_bound(l.begin(), l.end(), left[c]);
    if (lp != l.end())
      out[c] = lp - l.begin();
    else {
      const auto rp = std::lower_bound(r.begin(), r.end(), right[c]);
      if (rp != r.end()) out[c] = l.size() + (rp - r.begin());
    }
  }
  return out;
}
Row ProjectReverse(const Node& node, const Row& left, const Row& right) {
  auto out = EmptyReverse();
  for (size_t c = 0; c < kNumericalChains; ++c) {
    const auto& l = node.left_projection[c];
    const auto& r = node.right_projection[c];
    const auto rp = std::upper_bound(r.begin(), r.end(), right[c]);
    if (rp != r.begin())
      out[c] = l.size() + (rp - r.begin()) - 1;
    else {
      const auto lp = std::upper_bound(l.begin(), l.end(), left[c]);
      if (lp != l.begin()) out[c] = (lp - l.begin()) - 1;
    }
  }
  return out;
}
Row Reachable(const CompactSyncResult& result, const NodePtr& node, Ordinal event, bool completion) {
  if (!node->left) {
    auto out = EmptyForward(*node);
    for (size_t c = 0; c < kNumericalChains; ++c) {
      size_t low = 0, high = node->chains[c].size();
      while (low < high) {
        ++result.hierarchy_query_steps;
        const size_t middle = low + (high - low) / 2;
        if (LeafQuery(result, node->leaf, event, completion, node->chains[c][middle], c % 2))
          high = middle;
        else
          low = middle + 1;
      }
      out[c] = low;
    }
    return out;
  }
  auto left = EmptyForward(*node->left), right = EmptyForward(*node->right);
  if (event < node->left->length) {
    left = Reachable(result, node->left, event, completion);
    const auto entered = AcrossForward(*node, left, result);
    for (size_t c = 0; c < kNumericalChains; ++c)
      if (entered[c] < static_cast<int64_t>(node->right->chains[c].size()))
        for (size_t d = 0; d < kNumericalChains; ++d)
          right[d] = std::min(right[d], node->right->forward[c][entered[c]][d]);
  } else
    right = Reachable(result, node->right, event - node->left->length, completion);
  return ProjectForward(*node, left, right);
}
Row Ancestors(const CompactSyncResult& result, const NodePtr& node, Ordinal event, bool completion) {
  if (!node->left) {
    auto out = EmptyReverse();
    for (size_t c = 0; c < kNumericalChains; ++c) {
      size_t low = 0, high = node->chains[c].size();
      while (low < high) {
        ++result.hierarchy_query_steps;
        const size_t middle = low + (high - low) / 2;
        if (LeafQuery(result, node->leaf, node->chains[c][middle], c % 2, event, completion))
          low = middle + 1;
        else
          high = middle;
      }
      out[c] = static_cast<int64_t>(low) - 1;
    }
    return out;
  }
  auto left = EmptyReverse(), right = EmptyReverse();
  if (event >= node->left->length) {
    right = Ancestors(result, node->right, event - node->left->length, completion);
    const auto entered = AcrossReverse(*node, right, result);
    for (size_t c = 0; c < kNumericalChains; ++c)
      if (entered[c] >= 0)
        for (size_t d = 0; d < kNumericalChains; ++d)
          left[d] = std::max(left[d], node->left->reverse[c][entered[c]][d]);
  } else
    left = Ancestors(result, node->left, event, completion);
  return ProjectReverse(*node, left, right);
}
bool Query(const CompactSyncResult& result, const NodePtr& node, Ordinal source, bool sc, Ordinal target,
           bool tc) {
  ++result.hierarchy_query_steps;
  if (!node->left) return LeafQuery(result, node->leaf, source, sc, target, tc);
  const auto boundary = node->left->length;
  if (source < boundary && target < boundary) return Query(result, node->left, source, sc, target, tc);
  if (source >= boundary && target >= boundary)
    return Query(result, node->right, source - boundary, sc, target - boundary, tc);
  if (source >= boundary) return false;
  const auto reachable = Reachable(result, node->left, source, sc);
  const auto entered = AcrossForward(*node, reachable, result);
  const auto ancestors = Ancestors(result, node->right, target - boundary, tc);
  for (size_t c = 0; c < kNumericalChains; ++c)
    if (entered[c] <= ancestors[c]) return true;
  return false;
}
NumericalBoundary Point(const CompactSyncResult& result, const NodePtr& node, const NumericalCell& cell,
                        std::unordered_map<const Node*, NumericalBoundary>& cache) {
  const auto found = cache.find(node.get());
  if (found != cache.end()) return found->second;
  NumericalBoundary out;
  if (node->length) {
    if (!node->left)
      out = LeafBoundary(result, node->leaf, cell);
    else
      out = MergeNumericalBoundary(Point(result, node->left, cell, cache),
                                   Point(result, node->right, cell, cache), node->left->length);
  }
  cache.emplace(node.get(), out);
  return out;
}
}  // namespace

CompactSyncResult AnalyzeNumericalSync(CompactSyncInput input, const std::string& order) {
  CompactSyncResult result;
  result.input = std::move(input);
  auto hierarchy = std::make_shared<NumericalSyncHierarchy>();
  hierarchy->segment_offsets.push_back(0);
  for (size_t r = 0; r < result.input.segments.size(); ++r) {
    const auto& original = result.input.segments[r];
    result.segments.push_back(AnalyzeRotatingSegment(result.input, r, Scope(original), result));
    result.segments.back().repetitions = Repetitions(original);
    hierarchy->segment_offsets.push_back(
        NumericalAdd(hierarchy->segment_offsets.back(), Length(original.count, original.sites.size())));
  }
  const auto start = std::chrono::steady_clock::now();
  hierarchy->cells = InterfaceCells(result);
  std::vector<NodePtr> nodes;
  for (size_t r = 0; r < result.segments.size(); ++r) {
    auto leaf = BuildLeaf(result, hierarchy->cells, r);
    nodes.push_back(Repeat(leaf, result.segments[r].repetitions, result));
  }
  hierarchy->root = Compose(nodes, 0, nodes.size(), order, result);
  INTERNAL_CHECK(hierarchy->root->length == hierarchy->segment_offsets.back())
      << "Numerical repetition lost original occurrence coordinates";
  result.hierarchy = hierarchy;
  // Crossings remain owned by the hierarchy. A missing flat export is not an
  // empty answer; clients can audit bounded expansion without reanalysis.
  result.composition_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return result;
}
bool NumericalPrecedes(const CompactSyncResult& result, const Occurrence& a, bool sc, const Occurrence& b,
                       bool tc) {
  const auto source = Encode(result, a), target = Encode(result, b);
  if (source > target || (source == target && sc == tc)) return false;
  return Query(result, result.hierarchy->root, source, sc, target, tc);
}
PointBoundary QueryNumericalBoundary(const CompactSyncResult& result, size_t storage, int64_t slot,
                                     int64_t row, int64_t column) {
  std::unordered_map<const Node*, NumericalBoundary> cache;
  const auto selected = Point(result, result.hierarchy->root, {storage, slot, row, column}, cache);
  auto decode = [&](const std::optional<Ordinal>& ordinal) -> std::optional<Occurrence> {
    return ordinal ? std::optional<Occurrence>(Decode(result, *ordinal)) : std::nullopt;
  };
  PointBoundary out;
  out.first_writer = decode(selected.first_writer);
  out.last_writer = decode(selected.last_writer);
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = decode(selected.first_readers[p]);
    out.last_readers[p] = decode(selected.last_readers[p]);
  }
  return out;
}
std::optional<std::vector<Crossing>> ExportNumericalCrossings(const CompactSyncResult& result) {
  // Check the complete scope before exposing any records. A repeated DAG use
  // cannot be represented by the ordinary single-interval family coordinates.
  for (size_t r = 0; r < result.segments.size(); ++r) {
    const auto& segment = result.segments[r];
    if (segment.repetitions != 1 || segment.count != result.input.segments[r].count) return std::nullopt;
  }
  if (!result.hierarchy) return result.crossing_demands;
  std::vector<Crossing> out;
  ExportNodeCrossings(result, result.hierarchy->root, 0, out);
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
std::vector<Crossing> ExpandNumericalDemands(const CompactSyncResult& result, size_t maximum) {
  Ordinal count = 0;
  for (const auto& segment : result.input.segments)
    count = NumericalAdd(count, Length(segment.count, segment.sites.size()));
  CHECK(count <= maximum) << "Retained-demand audit exceeds its requested payload bound";
  std::vector<Crossing> out;
  if (result.hierarchy)
    ExpandNodeDemands(result, result.hierarchy->root, 0, out);
  else {
    out = result.crossing_demands;
    for (size_t r = 0; r < result.segments.size(); ++r) {
      const auto& segment = result.segments[r];
      for (const auto& demand : segment.demands)
        for (int64_t i = 0; i < segment.count - demand.distance; ++i)
          out.push_back({{r, i, demand.source}, {r, i + demand.distance, demand.target}});
    }
  }
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
  return out;
}
}  // namespace pypto::ir::sync
