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

#include "src/ir/transforms/utils/numerical_sync_merge.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
namespace pypto::ir::sync {
NumericalOrdinal NumericalAdd(NumericalOrdinal a, NumericalOrdinal b) {
  NumericalOrdinal value;
  CHECK(!__builtin_add_overflow(a, b, &value)) << "Numerical composition identity exceeds wide range";
  return value;
}
NumericalBoundary MergeNumericalBoundary(const NumericalBoundary& a, const NumericalBoundary& b,
                                         NumericalOrdinal offset) {
  auto shifted = [&](const std::optional<NumericalOrdinal>& value) -> std::optional<NumericalOrdinal> {
    return value ? std::optional<NumericalOrdinal>(NumericalAdd(offset, *value)) : std::nullopt;
  };
  NumericalBoundary out;
  out.first_writer = a.first_writer ? a.first_writer : shifted(b.first_writer);
  out.last_writer = b.last_writer ? shifted(b.last_writer) : a.last_writer;
  for (size_t p = 0; p < kPipeCount; ++p) {
    out.first_readers[p] = a.first_readers[p]
                               ? a.first_readers[p]
                               : (!a.first_writer ? shifted(b.first_readers[p]) : std::nullopt);
    out.last_readers[p] =
        b.last_readers[p] ? shifted(b.last_readers[p]) : (!b.last_writer ? a.last_readers[p] : std::nullopt);
  }
  return out;
}
std::set<NumericalEvent> NumericalPorts(const NumericalNode& node, size_t& visits) {
  std::set<NumericalEvent> ports;
  auto add = [&](const std::optional<NumericalOrdinal>& position) {
    ++visits;
    if (position) {
      ports.emplace(*position, false);
      ports.emplace(*position, true);
    }
  };
  for (const auto& boundary : node.storage) {
    add(boundary.first_writer);
    add(boundary.last_writer);
    for (size_t p = 0; p < kPipeCount; ++p) {
      add(boundary.first_readers[p]);
      add(boundary.last_readers[p]);
    }
  }
  for (size_t p = 0; p < kPipeCount; ++p) {
    add(node.first[p]);
    add(node.last[p]);
  }
  return ports;
}
void IndexNumericalPositions(NumericalNode& node) {
  for (size_t c = 0; c < kNumericalChains; ++c) {
    CHECK(node.chains[c].size() <= static_cast<size_t>(std::numeric_limits<int64_t>::max()))
        << "Numerical port chain exceeds signed rank range";
    for (size_t i = 0; i < node.chains[c].size(); ++i)
      node.positions.emplace(NumericalEvent{node.chains[c][i], c % 2 != 0}, NumericalPosition{c, i});
  }
}
void InvertNumericalRows(NumericalNode& node) {
  for (size_t d = 0; d < kNumericalChains; ++d) {
    node.reverse[d].resize(node.chains[d].size());
    for (auto& row : node.reverse[d]) row.fill(-1);
    for (size_t c = 0; c < kNumericalChains; ++c) {
      size_t source = 0;
      for (size_t target = 0; target < node.chains[d].size(); ++target) {
        while (source < node.chains[c].size() && node.forward[c][source][d] <= static_cast<int64_t>(target))
          ++source;
        node.reverse[d][target][c] = static_cast<int64_t>(source) - 1;
      }
    }
  }
}
void CountNumericalInterface(const NumericalNode& node, CompactSyncResult& result) {
  for (const auto& chain : node.chains) {
    result.interface_ports += chain.size();
    result.interface_entries += 2 * kNumericalChains * chain.size();
  }
}
namespace {
NumericalRow EmptyRow(const NumericalNode& node) {
  NumericalRow result;
  for (size_t c = 0; c < kNumericalChains; ++c) result[c] = static_cast<int64_t>(node.chains[c].size());
  return result;
}
struct Projection {
  std::array<std::vector<int64_t>, kNumericalChains> next, previous;
};
Projection Project(const NumericalNode& child, const NumericalNode& parent,
                   const std::array<std::vector<int64_t>, kNumericalChains>& selected, bool right) {
  Projection result;
  for (size_t c = 0; c < kNumericalChains; ++c) {
    const auto& keep = selected[c];
    const int64_t offset = right ? parent.left_projection[c].size() : 0;
    result.next[c].resize(child.chains[c].size() + 1, parent.chains[c].size());
    result.previous[c].resize(child.chains[c].size(), -1);
    size_t at = 0;
    for (size_t i = 0; i < child.chains[c].size(); ++i) {
      while (at < keep.size() && keep[at] < static_cast<int64_t>(i)) ++at;
      if (at < keep.size()) result.next[c][i] = offset + at;
      const auto preceding = at < keep.size() && keep[at] == static_cast<int64_t>(i) ? at + 1 : at;
      if (preceding) result.previous[c][i] = offset + preceding - 1;
    }
  }
  return result;
}
struct TwoMinimum {
  std::optional<std::pair<int64_t, size_t>> first, second;
  void Insert(int64_t value, size_t identity) {
    if (!first || value < first->first) {
      second = first;
      first = {{value, identity}};
    } else if (!second || value < second->first) {
      second = {{value, identity}};
    }
  }
  bool OtherAtMost(size_t identity, int64_t limit) const {
    const auto& chosen = first && first->second != identity ? first : second;
    return chosen && chosen->first <= limit;
  }
};
std::vector<NumericalCrossing> ReduceCrossings(const NumericalNode& a, const NumericalNode& b,
                                               const std::vector<NumericalCrossing>& all,
                                               CompactSyncResult& work) {
  std::vector<bool> redundant(all.size());
  for (size_t c = 0; c < kNumericalChains; ++c) {
    if (a.chains[c].empty()) continue;
    for (size_t d = 0; d < kNumericalChains; ++d) {
      if (b.chains[d].empty()) continue;
      std::vector<std::vector<size_t>> points(a.chains[c].size()), queries(a.chains[c].size());
      for (size_t i = 0; i < all.size(); ++i) {
        const auto& edge = all[i];
        if (edge.source.first == c && edge.target.first == d) points[edge.source.second].push_back(i);
        const auto threshold = a.forward[edge.source.first][edge.source.second][c];
        if (!edge.native && threshold < static_cast<int64_t>(a.chains[c].size()))
          queries[threshold].push_back(i);
      }
      TwoMinimum best;
      for (size_t rank = a.chains[c].size(); rank-- > 0;) {
        // Inclusive activation: ties enter before queries at this rank.
        for (size_t i : points[rank]) best.Insert(all[i].target.second, i);
        for (size_t i : queries[rank]) {
          ++work.crossing_tests;
          const auto& target = all[i].target;
          if (best.OtherAtMost(i, b.reverse[target.first][target.second][d])) redundant[i] = true;
        }
      }
    }
  }
  std::vector<NumericalCrossing> result;
  for (size_t i = 0; i < all.size(); ++i)
    if (!redundant[i]) result.push_back(all[i]);
  return result;
}
void BuildCrossIndex(NumericalNode& node) {
  const auto& a = *node.left;
  const auto& b = *node.right;
  // Rank buckets, not comparison sorting or an expanded relation table.
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t d = 0; d < kNumericalChains; ++d) {
      auto& index = node.crossing_index[c][d];
      std::vector<std::optional<int64_t>> destinations(a.chains[c].size()), sources(b.chains[d].size());
      for (const auto& edge : node.crossings) {
        if (edge.source.first != c || edge.target.first != d) continue;
        auto& target = destinations[edge.source.second];
        if (!target || static_cast<int64_t>(edge.target.second) < *target) target = edge.target.second;
        auto& source = sources[edge.target.second];
        if (!source || static_cast<int64_t>(edge.source.second) > *source) source = edge.source.second;
      }
      for (size_t i = 0; i < destinations.size(); ++i)
        if (destinations[i]) index.forward.emplace_back(i, *destinations[i]);
      for (size_t i = index.forward.size(); i-- > 1;)
        index.forward[i - 1].second = std::min(index.forward[i - 1].second, index.forward[i].second);
      for (size_t i = 0; i < sources.size(); ++i)
        if (sources[i]) index.reverse.emplace_back(i, *sources[i]);
      for (size_t i = 1; i < index.reverse.size(); ++i)
        index.reverse[i].second = std::max(index.reverse[i - 1].second, index.reverse[i].second);
    }
  }
}
}  // namespace
void BuildNumericalMergeOrder(const std::shared_ptr<NumericalNode>& node,
                              const std::set<NumericalEvent>& ports,
                              std::vector<NumericalCrossing> crossings) {
  CHECK(node && node->left && node->right && node->positions.empty())
      << "Numerical order merge requires a fresh parent and supplied children";
  const auto& a = node->left;
  const auto& b = node->right;
  CHECK(node->length == NumericalAdd(a->length, b->length)) << "Numerical order merge length mismatch";
  for (const auto& edge : crossings) {
    CHECK(edge.source.first < kNumericalChains && edge.target.first < kNumericalChains &&
          edge.source.second < a->chains[edge.source.first].size() &&
          edge.target.second < b->chains[edge.target.first].size())
        << "Numerical order crossing is outside supplied child ports";
  }
  node->crossings = std::move(crossings);
  const auto& all = node->crossings;
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t i = 0; i < a->chains[c].size(); ++i)
      if (ports.count({a->chains[c][i], c % 2 != 0})) {
        node->chains[c].push_back(a->chains[c][i]);
        node->left_projection[c].push_back(i);
      }
    for (size_t i = 0; i < b->chains[c].size(); ++i) {
      const auto ordinal = NumericalAdd(a->length, b->chains[c][i]);
      if (ports.count({ordinal, c % 2 != 0})) {
        node->chains[c].push_back(ordinal);
        node->right_projection[c].push_back(i);
      }
    }
  }
  IndexNumericalPositions(*node);
  CHECK(node->positions.size() == ports.size()) << "Numerical order projection omitted a supplied port";
  const auto left = Project(*a, *node, node->left_projection, false);
  const auto right = Project(*b, *node, node->right_projection, true);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    node->forward[c].resize(node->chains[c].size(), EmptyRow(*node));
    for (size_t i = 0; i < node->left_projection[c].size(); ++i)
      for (size_t d = 0; d < kNumericalChains; ++d)
        node->forward[c][i][d] = left.next[d][a->forward[c][node->left_projection[c][i]][d]];
    for (size_t i = 0; i < node->right_projection[c].size(); ++i)
      for (size_t d = 0; d < kNumericalChains; ++d)
        node->forward[c][i + node->left_projection[c].size()][d] =
            right.next[d][b->forward[c][node->right_projection[c][i]][d]];
    std::vector<std::vector<size_t>> activation(node->left_projection[c].size());
    for (size_t i = 0; i < all.size(); ++i) {
      const auto& source = all[i].source;
      const auto ancestor = a->reverse[source.first][source.second][c];
      if (ancestor < 0) continue;
      const auto last_source = left.previous[c][ancestor];
      if (last_source >= 0) activation[last_source].push_back(i);
    }
    auto active = EmptyRow(*node);
    for (size_t i = activation.size(); i-- > 0;) {
      for (size_t edge : activation[i]) {
        const auto& target = all[edge].target;
        for (size_t d = 0; d < kNumericalChains; ++d)
          active[d] = std::min(active[d], right.next[d][b->forward[target.first][target.second][d]]);
      }
      for (size_t d = 0; d < kNumericalChains; ++d)
        node->forward[c][i][d] = std::min(node->forward[c][i][d], active[d]);
    }
  }
  InvertNumericalRows(*node);
  BuildCrossIndex(*node);
}
std::shared_ptr<const NumericalNode> MergeNumericalNodes(std::shared_ptr<const NumericalNode> a,
                                                         std::shared_ptr<const NumericalNode> b,
                                                         CompactSyncResult& result) {
  if (!a->length) return b;
  if (!b->length) return a;
  auto node = std::make_shared<NumericalNode>();
  node->left = a;
  node->right = b;
  node->length = NumericalAdd(a->length, b->length);
  CHECK(a->storage.size() == b->storage.size()) << "Numerical children have incompatible storage frames";
  std::map<std::pair<NumericalEvent, NumericalEvent>, bool> edges;
  auto add = [&](NumericalOrdinal source, bool sc, NumericalOrdinal target, bool tc, bool native) {
    ++result.bridge_candidates;
    edges[{{source, sc}, {target, tc}}] |= native;
  };
  for (size_t cell = 0; cell < a->storage.size(); ++cell) {
    const auto& left = a->storage[cell];
    const auto& right = b->storage[cell];
    node->storage.push_back(MergeNumericalBoundary(left, right, a->length));
    if (left.last_writer) {
      if (right.first_writer) {
        const auto matrix = static_cast<size_t>(PipeType::M);
        // This is footprint protection, never a manufactured C->I edge. The
        // selected writers must be adjacent on the combined native M chain,
        // share the qualified physical accumulator, and have no intervening
        // read of it. Storage summaries retain such reads (including FIX).
        // Empty children preserve these facts; initialization clears the
        // target qualification even if its source site is otherwise protected.
        bool protected_pair = false;
        if (result.input.region_tree && a->last_m_qualified &&
            a->last_m_qualified == b->first_m_continuation &&
            *a->last_m_qualified == result.storage_classes.at(cell).storage &&
            left.last_writer == a->last[matrix] && right.first_writer == b->first[matrix]) {
          protected_pair = std::none_of(left.last_readers.begin(), left.last_readers.end(),
                                        [](const auto& value) { return value.has_value(); }) &&
                           std::none_of(right.first_readers.begin(), right.first_readers.end(),
                                        [](const auto& value) { return value.has_value(); });
        }
        if (!protected_pair) add(*left.last_writer, true, *right.first_writer, false, false);
      }
      for (const auto& reader : right.first_readers)
        if (reader) add(*left.last_writer, true, *reader, false, false);
    }
    if (right.first_writer)
      for (const auto& reader : left.last_readers)
        if (reader) add(*reader, true, *right.first_writer, false, false);
  }
  for (size_t p = 0; p < kPipeCount; ++p) {
    node->first[p] =
        a->first[p] ? a->first[p]
                    : (b->first[p] ? std::optional<NumericalOrdinal>(NumericalAdd(a->length, *b->first[p]))
                                   : std::nullopt);
    node->last[p] =
        b->last[p] ? std::optional<NumericalOrdinal>(NumericalAdd(a->length, *b->last[p])) : a->last[p];
    if (a->last[p] && b->first[p]) {
      add(*a->last[p], false, *b->first[p], false, true);
      add(*a->last[p], true, *b->first[p], true, true);
    }
  }
  const auto matrix = static_cast<size_t>(PipeType::M);
  node->first_m_continuation = a->first[matrix] ? a->first_m_continuation : b->first_m_continuation;
  node->last_m_qualified = b->last[matrix] ? b->last_m_qualified : a->last_m_qualified;
  std::vector<NumericalCrossing> all;
  for (const auto& [edge, native] : edges)
    all.push_back({a->positions.at(edge.first), b->positions.at(edge.second), native});
  // Parent ports are selected child ports, never synthetic completion events.
  const auto ports = NumericalPorts(*node, result.port_normalization_visits);
  // Demand proposal/reduction remains owned by this storage-analysis caller.
  // The supplied reduced crossings preserve the full order and the existing
  // compact query index; token clients instead supply their actual-use edges.
  BuildNumericalMergeOrder(node, ports, ReduceCrossings(*a, *b, all, result));
  CountNumericalInterface(*node, result);
  ++result.merge_nodes;
  return node;
}
}  // namespace pypto::ir::sync
