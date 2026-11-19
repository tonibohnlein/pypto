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

#include <cassert>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <type_traits>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"

namespace {
using pypto::ir::sync::CompactSyncBuilder;
using pypto::ir::sync::CompactSyncResult;
using pypto::ir::sync::CompactSyncView;
static_assert(!std::is_default_constructible_v<CompactSyncResult>);
static_assert(!std::is_copy_constructible_v<CompactSyncView>);
static_assert(!std::is_copy_assignable_v<CompactSyncView>);
static_assert(!std::is_move_assignable_v<CompactSyncView>);
static_assert(std::is_copy_constructible_v<CompactSyncResult>);
static_assert(std::is_const_v<std::remove_reference_t<decltype(std::declval<CompactSyncResult&>().State())>>);

void CheckPublishedLifetime() {
  // Returning a copy destroys both the builder and the first published owner.
  // The copy must retain the exact moved inventory, not borrowed member refs.
  auto retained = [] {
    CompactSyncBuilder builder;
    builder.State().input.hardware_target = "a2a3";
    builder.State().crossing_demands.push_back({{0, 2, 1}, {1, 0, 0}});
    const auto* inventory = builder.State().crossing_demands.data();
    auto published = std::move(builder).Publish();
    auto copy = published;
    assert(copy.State().crossing_demands.data() == inventory);
    assert(&copy.State() == &published.State());
    assert(&copy.Diagnostics() == &published.Diagnostics());
    return copy;
  }();
  assert(retained.State().input.hardware_target == "a2a3");
  assert(std::get<1>(retained.State().crossing_demands.front().first) == 2);
  CompactSyncBuilder another;
  another.State().input.hardware_target = "different construction";
  auto distinct = std::move(another).Publish();
  assert(&distinct.State() != &retained.State());
  assert(retained.State().input.hardware_target == "a2a3");
}
void CheckFamilyEndpointCoordinates() {
  using pypto::ir::sync::DecodeSyncFamilyEndpoint;
  using pypto::ir::sync::Occurrence;
  using pypto::ir::sync::PhysicalSyncFamily;
  using pypto::ir::sync::SyncEmptyAxes;
  using pypto::ir::sync::SyncEndpointKind;
  constexpr auto source = SyncEndpointKind::Source;
  constexpr auto target = SyncEndpointKind::Target;
  constexpr auto flat = SyncEmptyAxes::ContiguousVisits;
  constexpr auto scoped = SyncEmptyAxes::Singleton;
  PhysicalSyncFamily family{3, 7, 5, 11, 10, 20, 1, {}};
  assert(DecodeSyncFamilyEndpoint(family, 0, source, flat) == Occurrence(3, 10, 7));
  assert(DecodeSyncFamilyEndpoint(family, 0, target, scoped) == Occurrence(5, 20, 11));
  family.count = 3;
  assert(DecodeSyncFamilyEndpoint(family, 2, source, flat) == Occurrence(3, 12, 7));
  // Empty Cartesian axes mean singleton even at ordinal zero. The flat
  // convention must never silently make a missing scoped map executable.
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  assert(!DecodeSyncFamilyEndpoint(family, -1, source, flat));
  assert(!DecodeSyncFamilyEndpoint(family, 3, source, flat));
  family.count = 0;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, flat));
  family.count = -1;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, flat));

  // Independent nested enumeration: nonunit and unequal endpoint strides leave
  // holes and model reset-local coordinates. Frame legality is checked by the
  // production callers, not by this arithmetic-only decoder.
  for (int64_t outer = 1; outer <= 4; ++outer) {
    for (int64_t inner = 1; inner <= 5; ++inner) {
      family.count = outer * inner;
      family.axes = {{nullptr, outer, 12, 20}, {nullptr, inner, 2, 3}};
      int64_t ordinal = 0;
      for (int64_t i = 0; i < outer; ++i) {
        for (int64_t j = 0; j < inner; ++j, ++ordinal) {
          assert(DecodeSyncFamilyEndpoint(family, ordinal, source, scoped) ==
                 Occurrence(3, 10 + 12 * i + 2 * j, 7));
          assert(DecodeSyncFamilyEndpoint(family, ordinal, target, scoped) ==
                 Occurrence(5, 20 + 20 * i + 3 * j, 11));
          assert(DecodeSyncFamilyEndpoint(family, ordinal, source, flat) ==
                 DecodeSyncFamilyEndpoint(family, ordinal, source, scoped));
        }
      }
    }
  }
  family.count = 2;
  family.axes = {{nullptr, 2, 0, 3}};
  assert(DecodeSyncFamilyEndpoint(family, 1, source, scoped) == Occurrence(3, 10, 7));
  assert(DecodeSyncFamilyEndpoint(family, 1, target, scoped) == Occurrence(5, 23, 11));
  family.axes.front().count = 3;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  family.axes.front().count = 0;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  family.axes.front().count = -1;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  family.axes.front().count = 2;
  family.axes.front().source_stride = -1;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  family.axes.front().source_stride = 1;
  family.source_first = -1;
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
  family.source_first = std::numeric_limits<int64_t>::max();
  assert(!DecodeSyncFamilyEndpoint(family, 1, source, scoped));
  family.axes.clear();
  assert(!DecodeSyncFamilyEndpoint(family, 1, source, flat));
  family.source_first = 10;
  family.count = std::numeric_limits<int64_t>::max();
  family.axes = {{nullptr, family.count, 1, 1}, {nullptr, family.count, 1, 1}, {nullptr, family.count, 1, 1}};
  assert(!DecodeSyncFamilyEndpoint(family, 0, source, scoped));
}
using pypto::ir::sync::BuildNumericalMergeOrder;
using pypto::ir::sync::IndexNumericalPositions;
using pypto::ir::sync::InvertNumericalRows;
using pypto::ir::sync::kNumericalChains;
using pypto::ir::sync::NumericalCrossing;
using pypto::ir::sync::NumericalEvent;
using pypto::ir::sync::NumericalNode;
using pypto::ir::sync::NumericalOrderQuery;
using pypto::ir::sync::NumericalOrdinal;
using pypto::ir::sync::NumericalRow;
using pypto::ir::sync::QueryNumericalOrder;
using Graph = std::vector<std::vector<bool>>;
using Node = std::shared_ptr<const NumericalNode>;

Graph Closure(Graph graph) {
  for (size_t i = 0; i < graph.size(); ++i) graph[i][i] = true;
  for (size_t k = 0; k < graph.size(); ++k)
    for (size_t i = 0; i < graph.size(); ++i)
      for (size_t j = 0; j < graph.size(); ++j) graph[i][j] = graph[i][j] || (graph[i][k] && graph[k][j]);
  return graph;
}
struct Fixture {
  std::mt19937 random;
  std::vector<Graph> leaf_graphs;
  size_t comparisons = 0, query_steps = 0;
  explicit Fixture(uint32_t seed) : random(seed) {}

  std::pair<Node, Graph> Leaf(size_t count) {
    Graph graph(2 * count, std::vector<bool>(2 * count));
    auto node = std::make_shared<NumericalNode>();
    node->length = count;
    node->leaf = leaf_graphs.size();
    std::vector<size_t> pipes;
    for (size_t i = 0; i < count; ++i) {
      const auto pipe = random() % 3;
      pipes.push_back(pipe);
      graph[2 * i][2 * i + 1] = true;
      for (size_t j = 0; j < i; ++j) {
        if (pipe == pipes[j]) {
          graph[2 * j][2 * i] = true;
          graph[2 * j + 1][2 * i + 1] = true;
        }
        if (random() % 3 == 0) graph[2 * j + 1][2 * i] = true;
      }
      node->chains[2 * pipe].push_back(i);
      node->chains[2 * pipe + 1].push_back(i);
    }
    auto closure = Closure(graph);
    leaf_graphs.push_back(closure);
    IndexNumericalPositions(*node);
    for (size_t c = 0; c < kNumericalChains; ++c) {
      for (auto source : node->chains[c]) {
        NumericalRow row;
        for (size_t d = 0; d < kNumericalChains; ++d) {
          row[d] = node->chains[d].size();
          for (size_t j = 0; j < node->chains[d].size(); ++j)
            if (closure[2 * source + c % 2][2 * node->chains[d][j] + d % 2]) {
              row[d] = j;
              break;
            }
        }
        node->forward[c].push_back(row);
      }
    }
    InvertNumericalRows(*node);
    return {node, graph};
  }

  std::pair<Node, Graph> Merge(const std::pair<Node, Graph>& a, const std::pair<Node, Graph>& b,
                               bool select_ports) {
    auto parent = std::make_shared<NumericalNode>();
    parent->left = a.first;
    parent->right = b.first;
    parent->length = a.first->length + b.first->length;
    const size_t offset = a.second.size();
    Graph graph(offset + b.second.size(), std::vector<bool>(offset + b.second.size()));
    for (size_t i = 0; i < offset; ++i)
      for (size_t j = 0; j < offset; ++j) graph[i][j] = a.second[i][j];
    for (size_t i = 0; i < b.second.size(); ++i)
      for (size_t j = 0; j < b.second.size(); ++j) graph[offset + i][offset + j] = b.second[i][j];
    std::vector<NumericalCrossing> edges;
    std::set<NumericalEvent> ports;
    for (const auto& [event, position] : a.first->positions) {
      if (!select_ports || !position.second ||
          position.second + 1 == a.first->chains[position.first].size() || random() % 2)
        ports.insert(event);
      for (const auto& [target, other] : b.first->positions) {
        if (random() % 7) continue;
        edges.push_back({position, other, false});
        graph[2 * event.first + event.second][offset + 2 * target.first + target.second] = true;
      }
    }
    for (const auto& [event, position] : b.first->positions)
      if (!select_ports || !position.second ||
          position.second + 1 == b.first->chains[position.first].size() || random() % 2)
        ports.emplace(a.first->length + event.first, event.second);
    // Concatenated chain rows require the native links between both children.
    // The order service consumes these links; it must not invent them.
    for (size_t chain = 0; chain < kNumericalChains; ++chain) {
      const auto& left = a.first->chains[chain];
      const auto& right = b.first->chains[chain];
      if (left.empty() || right.empty()) continue;
      edges.push_back({{chain, left.size() - 1}, {chain, 0}, true});
      graph[2 * left.back() + chain % 2][offset + 2 * right.front() + chain % 2] = true;
    }
    BuildNumericalMergeOrder(parent, ports, edges);
    // The order service does not generate, remove, or reinterpret handoffs.
    assert(parent->crossings.size() == edges.size());
    assert(parent->storage.empty());
    return {parent, graph};
  }

  void Check(const std::pair<Node, Graph>& composed) {
    const auto closure = Closure(composed.second);
    const NumericalOrderQuery query{
        [&](size_t leaf, NumericalOrdinal a, bool ac, NumericalOrdinal b, bool bc) {
          return leaf_graphs.at(leaf).at(2 * a + ac).at(2 * b + bc);
        },
        query_steps};
    for (size_t a = 0; a < closure.size(); ++a)
      for (size_t b = 0; b < closure.size(); ++b) {
        assert(QueryNumericalOrder(query, composed.first, a / 2, a % 2, b / 2, b % 2) == closure[a][b]);
        ++comparisons;
      }
    // All exported rows must agree with the independent flattened graph.
    for (const auto& [a, pa] : composed.first->positions)
      for (const auto& [b, pb] : composed.first->positions)
        assert((composed.first->forward[pa.first][pa.second][pb.first] <= static_cast<int64_t>(pb.second)) ==
               closure[2 * a.first + a.second][2 * b.first + b.second]);
  }
};
}  // namespace

int main() {
  CheckPublishedLifetime();
  CheckFamilyEndpointCoordinates();
  size_t comparisons = 0;
  for (uint32_t seed = 0; seed < 2000; ++seed) {
    Fixture fixture(seed);
    const auto a = fixture.Leaf(1 + seed % 4);
    const auto b = fixture.Leaf(1 + (seed / 4) % 4);
    const auto c = fixture.Leaf(1 + (seed / 16) % 4);
    const auto ab = fixture.Merge(a, b, true);
    fixture.Check(ab);
    fixture.Check(fixture.Merge(ab, c, true));
    const auto bc = fixture.Merge(b, c, true);
    fixture.Check(bc);
    fixture.Check(fixture.Merge(a, bc, false));
    comparisons += fixture.comparisons;
  }
  std::cout << "2000 supplied-order graphs, " << comparisons << " event-query comparisons passed\n";
}
