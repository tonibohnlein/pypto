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

#include "src/ir/transforms/utils/conservative_sync_completion.h"

#include <algorithm>
#include <cstddef>
#include <deque>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"

namespace pypto::ir::sync {
namespace {

size_t PipeIndex(PipeType pipe) {
  INTERNAL_CHECK(pipe != PipeType::ALL) << "Internal error: PipeType::ALL cannot index single pipe lane";
  const int index = static_cast<int>(pipe);
  INTERNAL_CHECK(index >= 0 && static_cast<size_t>(index) < kPhysicalPipeCount)
      << "Internal error: unexpected PipeType value " << index;
  return static_cast<size_t>(index);
}

void ApplyEdgeToMatrix(std::array<PipeMask, kPhysicalPipeCount>& matrix, const CompletionSyncEdge& edge) {
  const PipeMask src_mask = PipeToMask(edge.src);
  const PipeMask dst_mask = PipeToMask(edge.dst);
  if (src_mask == 0 || dst_mask == 0) return;
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    const PipeMask before = matrix[pipe];
    PipeMask after = before;
    if ((before & src_mask) != 0) after |= dst_mask;
    if (edge.bidirectional && (before & dst_mask) != 0) after |= src_mask;
    matrix[pipe] = after;
  }
}

void RelayMatrix(std::array<PipeMask, kPhysicalPipeCount>& target,
                 const std::array<PipeMask, kPhysicalPipeCount>& relay) {
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    const PipeMask reached = target[pipe];
    PipeMask next = reached;
    for (size_t mid = 0; mid < kPhysicalPipeCount; ++mid) {
      if ((reached & (PipeMask{1} << mid)) != 0) next |= relay[mid];
    }
    target[pipe] = next;
  }
}

CompletionFacts EvaluateVertex(const CompletionVertex& vertex, const CompletionFacts& incoming) {
  CompletionFacts out = incoming;
  out.ApplySyncEdges(vertex.pre_syncs);
  if (vertex.nested_summary.has_value()) {
    out.ApplySummary(*vertex.nested_summary);
  }
  if (vertex.write_pipes != 0) {
    out.ApplyWritePipes(vertex.write_pipes);
  }
  out.ApplySyncEdges(vertex.post_syncs);
  return out;
}

}  // namespace

CompletionSummary CompletionSummary::Identity() {
  CompletionSummary summary;
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    summary.relay[pipe] = PipeMask{1} << pipe;
    summary.post_write_safe[pipe] = PipeMask{1} << pipe;
  }
  summary.has_bypass_path = true;
  return summary;
}

bool CompletionSummary::IsRelayed(PipeType from, PipeType to) const {
  if (from == PipeType::ALL || to == PipeType::ALL) {
    const PipeMask src_mask = PipeToMask(from);
    const PipeMask dst_mask = PipeToMask(to);
    for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
      if ((src_mask & (PipeMask{1} << pipe)) == 0) continue;
      if ((relay[pipe] & dst_mask) != dst_mask) return false;
    }
    return src_mask != 0 && dst_mask != 0;
  }
  return (relay[PipeIndex(from)] & PipeToMask(to)) != 0;
}

bool CompletionSummary::IsPostWriteSafe(PipeType producer, PipeType consumer) const {
  const PipeMask prod_mask = PipeToMask(producer);
  const PipeMask cons_mask = PipeToMask(consumer);
  if (prod_mask == 0 || cons_mask == 0) return false;
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    if ((prod_mask & (PipeMask{1} << pipe)) == 0) continue;
    if ((active & (PipeMask{1} << pipe)) == 0) continue;
    if ((post_write_safe[pipe] & cons_mask) != cons_mask) return false;
  }
  return true;
}

CompletionFacts CompletionFacts::AtEntry() {
  CompletionFacts facts;
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    facts.relay[pipe] = PipeMask{1} << pipe;
    facts.post_write_safe[pipe] = kAllPhysicalPipesMask;
  }
  facts.has_bypass_path = true;
  return facts;
}

void CompletionFacts::MergeFrom(const CompletionFacts& other) {
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    relay[pipe] &= other.relay[pipe];
    const PipeMask bit = PipeMask{1} << pipe;
    const bool lhs_active = (active & bit) != 0;
    const bool rhs_active = (other.active & bit) != 0;
    if (lhs_active && rhs_active) {
      post_write_safe[pipe] &= other.post_write_safe[pipe];
    } else if (rhs_active) {
      post_write_safe[pipe] = other.post_write_safe[pipe];
    }
  }
  active |= other.active;
  has_bypass_path = has_bypass_path || other.has_bypass_path;
}

void CompletionFacts::ApplySyncEdges(const std::vector<CompletionSyncEdge>& edges) {
  for (const auto& edge : edges) {
    if (PipeToMask(edge.src) == 0 || PipeToMask(edge.dst) == 0) continue;
    has_bypass_path = false;
    ApplyEdgeToMatrix(relay, edge);
    ApplyEdgeToMatrix(post_write_safe, edge);
  }
}

void CompletionFacts::ApplyWritePipes(PipeMask writes) {
  const PipeMask physical = writes & kAllPhysicalPipesMask;
  if (physical == 0) return;
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    const PipeMask bit = PipeMask{1} << pipe;
    if ((physical & bit) == 0) continue;
    active |= bit;
    post_write_safe[pipe] = bit;
  }
}

void CompletionFacts::ApplySummary(const CompletionSummary& summary) {
  RelayMatrix(relay, summary.relay);
  for (size_t pipe = 0; pipe < kPhysicalPipeCount; ++pipe) {
    const PipeMask bit = PipeMask{1} << pipe;
    if ((active & bit) != 0) {
      PipeMask next = post_write_safe[pipe];
      for (size_t mid = 0; mid < kPhysicalPipeCount; ++mid) {
        if ((post_write_safe[pipe] & (PipeMask{1} << mid)) != 0) next |= summary.relay[mid];
      }
      post_write_safe[pipe] = next;
    }
    if ((summary.active & bit) != 0) {
      if ((active & bit) != 0) {
        post_write_safe[pipe] &= summary.post_write_safe[pipe];
      } else {
        post_write_safe[pipe] = summary.post_write_safe[pipe];
      }
    }
  }
  active |= summary.active;
  has_bypass_path = has_bypass_path && summary.has_bypass_path;
}

CompletionSummary CompletionFacts::ToSummary() const {
  CompletionSummary summary;
  summary.active = active;
  summary.relay = relay;
  summary.post_write_safe = post_write_safe;
  summary.has_bypass_path = has_bypass_path;
  return summary;
}

CompletionTransferResult SolveCompletionTransfer(const std::vector<CompletionVertex>& vertices,
                                                 const std::vector<std::pair<size_t, size_t>>& edges, size_t entry,
                                                 size_t exit) {
  const size_t n = vertices.size();
  INTERNAL_CHECK(entry < n && exit < n) << "Internal error: completion transfer entry/exit vertex out of bounds";

  std::vector<std::vector<size_t>> succs(n);
  std::vector<std::vector<size_t>> preds(n);
  std::vector<bool> self_loop(n, false);
  for (const auto& [from, to] : edges) {
    INTERNAL_CHECK(from < n && to < n) << "Internal error: completion transfer edge endpoint out of bounds";
    if (from == to) {
      self_loop[from] = true;
    }
    succs[from].push_back(to);
    preds[to].push_back(from);
  }
  for (size_t u = 0; u < n; ++u) {
    std::sort(succs[u].begin(), succs[u].end());
    succs[u].erase(std::unique(succs[u].begin(), succs[u].end()), succs[u].end());
    std::sort(preds[u].begin(), preds[u].end());
    preds[u].erase(std::unique(preds[u].begin(), preds[u].end()), preds[u].end());
  }

  // Forward iterative DFS from `entry` to find reachable vertices and post-order.
  std::vector<bool> reachable(n, false);
  std::vector<size_t> post_order;
  post_order.reserve(n);
  {
    std::vector<std::pair<size_t, size_t>> stack;
    reachable[entry] = true;
    stack.emplace_back(entry, 0);
    while (!stack.empty()) {
      auto& [u, next_idx] = stack.back();
      if (next_idx < succs[u].size()) {
        const size_t v = succs[u][next_idx++];
        if (!reachable[v]) {
          reachable[v] = true;
          stack.emplace_back(v, 0);
        }
      } else {
        post_order.push_back(u);
        stack.pop_back();
      }
    }
  }

  // Kosaraju second pass on transpose graph restricted to reachable vertices -> topological SCC order.
  std::vector<size_t> scc_id(n, n);
  std::vector<std::vector<size_t>> sccs;
  {
    std::vector<bool> visited_t(n, false);
    std::vector<size_t> dfs_stack;
    for (auto it = post_order.rbegin(); it != post_order.rend(); ++it) {
      const size_t seed = *it;
      if (visited_t[seed]) continue;
      const size_t cid = sccs.size();
      std::vector<size_t> comp;
      visited_t[seed] = true;
      dfs_stack.push_back(seed);
      while (!dfs_stack.empty()) {
        const size_t u = dfs_stack.back();
        dfs_stack.pop_back();
        scc_id[u] = cid;
        comp.push_back(u);
        for (const size_t p : preds[u]) {
          if (reachable[p] && !visited_t[p]) {
            visited_t[p] = true;
            dfs_stack.push_back(p);
          }
        }
      }
      std::sort(comp.begin(), comp.end());
      sccs.push_back(std::move(comp));
    }
  }

  CompletionTransferResult result;
  std::vector<CompletionFacts> out_facts(n, CompletionFacts::AtEntry());
  result.work.sccs = sccs.size();

  std::vector<bool> visited(n, false);
  auto compute_in = [&](size_t u, CompletionFacts& dst) {
    bool has_in = (u == entry);
    if (has_in) dst = CompletionFacts::AtEntry();
    for (const size_t p : preds[u]) {
      if (!reachable[p] || !visited[p]) continue;
      if (!has_in) {
        dst = out_facts[p];
        has_in = true;
      } else {
        dst.MergeFrom(out_facts[p]);
      }
    }
    return has_in;
  };

  for (size_t cid = 0; cid < sccs.size(); ++cid) {
    const auto& comp = sccs[cid];
    if (comp.size() == 1 && !self_loop[comp[0]]) {
      const size_t u = comp[0];
      CompletionFacts in_u;
      const bool has_in = compute_in(u, in_u);
      INTERNAL_CHECK(has_in) << "Internal error: reachable topological vertex missing visited predecessor";
      out_facts[u] = EvaluateVertex(vertices[u], in_u);
      visited[u] = true;
      continue;
    }

    ++result.work.non_trivial_sccs;
    std::deque<size_t> queue;
    std::vector<bool> in_queue(n, false);
    for (const size_t u : comp) {
      bool ready = (u == entry);
      if (!ready) {
        for (const size_t p : preds[u]) {
          if (reachable[p] && visited[p]) {
            ready = true;
            break;
          }
        }
      }
      if (ready) {
        queue.push_back(u);
        in_queue[u] = true;
      }
    }
    while (!queue.empty()) {
      const size_t u = queue.front();
      queue.pop_front();
      in_queue[u] = false;
      CompletionFacts in_u;
      if (!compute_in(u, in_u)) continue;
      const CompletionFacts out_u = EvaluateVertex(vertices[u], in_u);
      if (!visited[u] || out_u != out_facts[u]) {
        visited[u] = true;
        out_facts[u] = out_u;
        for (const size_t v : succs[u]) {
          if (scc_id[v] == cid && !in_queue[v]) {
            queue.push_back(v);
            in_queue[v] = true;
          }
        }
      }
    }
  }

  result.summary = reachable[exit] ? out_facts[exit].ToSummary() : CompletionSummary::Identity();
  return result;
}

}  // namespace pypto::ir::sync
