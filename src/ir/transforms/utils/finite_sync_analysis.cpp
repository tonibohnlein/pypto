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

#include "pypto/ir/transforms/utils/finite_sync_analysis.h"

#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto::ir::sync {
namespace {
void Join(RankRow& target, const RankRow& source) {
  for (size_t p = 0; p < target.size(); ++p) target[p] = std::max(target[p], source[p]);
}
struct Frontier {
  std::optional<size_t> writer;
  std::vector<size_t> readers;
};
}  // namespace

FiniteSyncResult AnalyzeFiniteSync(const FiniteSyncInput& input) {
  FiniteSyncResult result;
  result.exact_accesses = input.exact_accesses;
  const size_t n = input.occurrences.size();
  result.ranks.resize(n);
  result.starts.resize(n);
  result.completions.resize(n);
  std::unordered_map<Cell, Frontier> frontiers;
  for (Cell cell : input.covered_cells) {
    result.storage.try_emplace(cell);
    frontiers.try_emplace(cell);
  }
  std::vector<std::unordered_set<size_t>> outgoing(n);
  std::vector<std::vector<size_t>> native(n);
  std::array<size_t, kPipeCount> counts{};
  for (const auto& occurrence : input.occurrences) {
    CHECK(occurrence.pipe >= PipeType::MTE1 && occurrence.pipe < PipeType::ALL)
        << "Finite synchronization requires a concrete payload pipe";
    result.pipes.push_back(occurrence.pipe);
  }
  auto emit = [&](size_t source, size_t target) {
    ++result.work.witnesses;
    outgoing[source].insert(target);
  };
  for (size_t b = 0; b < n; ++b) {
    const auto& occurrence = input.occurrences[b];
    // RMW queries the old frontier once, then installs the new writer.
    std::unordered_map<Cell, bool> accesses;
    for (Cell cell : occurrence.reads) accesses.try_emplace(cell, false);
    for (Cell cell : occurrence.writes) accesses[cell] = true;
    result.work.access_entries += occurrence.reads.size() + occurrence.writes.size();
    for (const auto& access : accesses) {
      CHECK(frontiers.count(access.first)) << "Finite synchronization access outside covered storage domain";
    }
    if (!occurrence.present) continue;
    const size_t pipe = static_cast<size_t>(occurrence.pipe);
    result.ranks[b] = ++counts[pipe];
    for (const auto& [cell, writes] : accesses) {
      ++result.work.cell_visits;
      auto& frontier = frontiers.at(cell);
      auto& boundary = result.storage.at(cell);
      if (frontier.writer) emit(*frontier.writer, b);
      if (writes) {
        for (size_t reader : frontier.readers) {
          ++result.work.reader_visits;
          emit(reader, b);
        }
        frontier.readers.clear();
        frontier.writer = b;
        if (!boundary.first_writer) boundary.first_writer = b;
        boundary.last_writer = b;
        boundary.last_readers.fill(std::nullopt);
      } else {
        frontier.readers.push_back(b);
        if (!boundary.first_writer && !boundary.first_readers[pipe]) boundary.first_readers[pipe] = b;
        boundary.last_readers[pipe] = b;
      }
    }
  }
  auto validate = [&](const Demand& edge) {
    CHECK(edge.first < edge.second && edge.second < n)
        << "Finite synchronization prerequisites must have valid forward occurrence indices";
    return result.ranks[edge.first] != 0 && result.ranks[edge.second] != 0;
  };
  for (const auto& edge : input.prerequisites) {
    if (validate(edge)) emit(edge.first, edge.second);
  }
  for (const auto& edge : input.native_order) {
    if (validate(edge)) native[edge.second].push_back(edge.first);
  }
  // Reverse-source distribution orders every incoming list in O(n+g) work.
  // Hash iteration cannot affect output: each target receives sources in the
  // deterministic outer-loop order, without per-target sorting or full scans.
  std::vector<std::vector<size_t>> incoming(n);
  for (size_t a = n; a-- > 0;) {
    for (size_t b : outgoing[a]) incoming[b].push_back(a);
  }
  std::array<std::optional<size_t>, kPipeCount> previous{};
  for (size_t b = 0; b < n; ++b) {
    if (!result.ranks[b]) continue;
    const size_t q = static_cast<size_t>(result.pipes[b]);
    auto& start = result.starts[b];
    auto& completion = result.completions[b];
    if (previous[q]) start = result.starts[*previous[q]];
    // Native guarantees participate before reduction, including paths through
    // other cells. Leaving them until emission could retain redundant demands.
    for (size_t a : native[b]) Join(start, result.completions[a]);
    for (size_t a : incoming[b]) {
      result.generators.emplace_back(a, b);
      ++result.work.candidate_tests;
      const size_t p = static_cast<size_t>(result.pipes[a]);
      if (start[kPipeCount + p] < result.ranks[a]) {
        result.demands.emplace_back(a, b);
        Join(start, result.completions[a]);
        ++result.work.demand_joins;
      }
    }
    start[q] = result.ranks[b];
    completion = start;
    // Native completion order is not completion-before-next-start order.
    if (previous[q]) Join(completion, result.completions[*previous[q]]);
    completion[kPipeCount + q] = result.ranks[b];
    previous[q] = b;
  }
  return result;
}

bool FiniteSyncResult::Precedes(size_t source, bool source_completion, size_t target,
                                bool target_completion) const {
  CHECK(source < ranks.size() && target < ranks.size()) << "Invalid finite synchronization query occurrence";
  if (!ranks[source] || !ranks[target]) return false;
  if (source == target && source_completion == target_completion) return false;
  const auto& row = target_completion ? completions[target] : starts[target];
  const size_t column = static_cast<size_t>(pipes[source]) + (source_completion ? kPipeCount : 0);
  return row[column] >= ranks[source];
}
std::optional<StorageBoundary> FiniteSyncResult::Boundary(Cell cell) const {
  const auto found = storage.find(cell);
  if (found == storage.end()) return std::nullopt;
  return found->second;
}
}  // namespace pypto::ir::sync
