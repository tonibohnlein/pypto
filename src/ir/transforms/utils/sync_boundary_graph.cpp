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

#include "src/ir/transforms/utils/sync_boundary_graph.h"

#include <array>
#include <set>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto::ir::sync {
size_t SyncBoundaryGraph::Node(const Event& e) {
  auto [found, inserted] = nodes_.emplace(e, nodes_.size());
  if (inserted) graph_.emplace_back();
  return found->second;
}
bool SyncBoundaryGraph::Path(size_t a, size_t b, std::optional<std::pair<size_t, size_t>> skip) const {
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
void SyncBoundaryGraph::Add(const Event& a, const Event& b) {
  CHECK(nodes_.count(a) && nodes_.count(b)) << "Crossing endpoint outside retained boundary ports";
  const auto source = Node(a), target = Node(b);
  graph_[source].push_back(target);
}
size_t SyncBoundaryGraph::Edges() const {
  size_t result = 0;
  for (const auto& edges : graph_) result += edges.size();
  return result;
}
SyncBoundaryGraph::SyncBoundaryGraph(const std::vector<CompactSegmentResult>& segments)
    : ports_(segments.size()) {
  for (size_t r = 0; r < segments.size(); ++r) {
    const auto& s = segments[r];
    if (!s.count) continue;
    for (int64_t i : std::set<int64_t>{0, s.count - 1})
      for (size_t p = 0; p < s.pipes.size(); ++p)
        for (bool c : {false, true}) Node({{r, i, p}, c});
  }
  for (const auto& [event, index] : nodes_) ports_[std::get<0>(event.first)].emplace_back(event, index);
  for (const auto& child : ports_)
    for (const auto& [a, ai] : child)
      for (const auto& [b, bi] : child) {
        ++pair_tests;
        auto [ar, av, as] = a.first;
        auto [br, bv, bs] = b.first;
        if (segments[ar].Precedes(av, as, a.second, bv, bs, b.second)) graph_[ai].push_back(bi);
      }
  std::array<std::optional<Occurrence>, kPipeCount> previous{};
  for (size_t r = 0; r < segments.size(); ++r) {
    const auto& s = segments[r];
    if (!s.count) continue;
    std::array<std::optional<size_t>, kPipeCount> first{}, last{};
    for (size_t p = 0; p < s.pipes.size(); ++p) {
      if (!first[s.pipes[p]]) first[s.pipes[p]] = p;
      last[s.pipes[p]] = p;
    }
    for (size_t p = 0; p < kPipeCount; ++p)
      if (first[p]) {
        if (previous[p])
          for (bool c : {false, true}) Add({*previous[p], c}, {{r, 0, *first[p]}, c});
        previous[p] = Occurrence{r, s.count - 1, *last[p]};
      }
  }
}
bool SyncBoundaryGraph::Precedes(const std::vector<CompactSegmentResult>& segments, const Occurrence& a,
                                 bool sc, const Occurrence& b, bool tc) const {
  const auto [ar, av, as] = a;
  const auto [br, bv, bs] = b;
  std::vector<size_t> starts, ends;
  for (const auto& [event, node] : ports_.at(ar)) {
    const auto [r, v, s] = event.first;
    if ((event.first == a && event.second == sc) || segments[ar].Precedes(av, as, sc, v, s, event.second))
      starts.push_back(node);
  }
  for (const auto& [event, node] : ports_.at(br)) {
    const auto [r, v, s] = event.first;
    if ((event.first == b && event.second == tc) || segments[br].Precedes(v, s, event.second, bv, bs, tc))
      ends.push_back(node);
  }
  // One multi-source traversal is equivalent to the former starts x ends
  // searches. Child queries and boundary directory selection are charged
  // separately; graph work is O(P+E) per cross-region query.
  std::vector<bool> seen(graph_.size()), target(graph_.size());
  for (auto node : ends) target[node] = true;
  for (auto node : starts) seen[node] = true;
  while (!starts.empty()) {
    const auto source = starts.back();
    starts.pop_back();
    for (auto next : graph_[source]) {
      if (target[next]) return true;
      if (!seen[next]) {
        seen[next] = true;
        starts.push_back(next);
      }
    }
  }
  return false;
}
}  // namespace pypto::ir::sync
