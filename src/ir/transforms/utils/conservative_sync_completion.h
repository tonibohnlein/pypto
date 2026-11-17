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

#ifndef SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_COMPLETION_H_
#define SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_COMPLETION_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/ir/pipe.h"

namespace pypto::ir::sync {

// Number of physical pipeline lanes in PipeType (MTE1..FIX, excluding ALL).
inline constexpr size_t kPhysicalPipeCount = 7;
using PipeMask = uint32_t;
inline constexpr PipeMask kAllPhysicalPipesMask = (PipeMask{1} << kPhysicalPipeCount) - PipeMask{1};

// Return single-bit or all-pipes mask for a PipeType lane.
inline constexpr PipeMask PipeToMask(PipeType pipe) {
  if (pipe == PipeType::ALL) return kAllPhysicalPipesMask;
  const int index = static_cast<int>(pipe);
  if (index < 0 || static_cast<size_t>(index) >= kPhysicalPipeCount) return 0;
  return PipeMask{1} << static_cast<size_t>(index);
}

// Directed or bidirectional synchronization edge executed at a completion vertex.
struct CompletionSyncEdge {
  PipeType src = PipeType::S;
  PipeType dst = PipeType::S;
  bool bidirectional = true;
};

// Region completion summary corresponding to Draft Section 8.3 (A, E, S, t):
// - active (A): pipes that may execute a memory access inside the region.
// - post_write_safe (E): for each p in active, pipes q guaranteed ordered after every
//   access on p along all paths in the region that touch p.
// - relay (S): for each pipe p, pipes q guaranteed ordered after p along all paths from
//   entry to exit (every path includes a sync chain p ->+ q or p == q).
// - has_bypass_path (t): whether a path from entry to exit exists with zero sync edges.
struct CompletionSummary {
  PipeMask active = 0;
  std::array<PipeMask, kPhysicalPipeCount> post_write_safe{};
  std::array<PipeMask, kPhysicalPipeCount> relay{};
  bool has_bypass_path = false;

  static CompletionSummary Identity();

  bool IsRelayed(PipeType from, PipeType to) const;
  bool IsPostWriteSafe(PipeType producer, PipeType consumer) const;
  bool operator==(const CompletionSummary& other) const {
    return active == other.active && post_write_safe == other.post_write_safe && relay == other.relay &&
           has_bypass_path == other.has_bypass_path;
  }
  bool operator!=(const CompletionSummary& other) const { return !(*this == other); }
};

// Monotone CFG completion state at a vertex over the Section 8.3 lattice.
struct CompletionFacts {
  PipeMask active = 0;
  std::array<PipeMask, kPhysicalPipeCount> post_write_safe{};
  std::array<PipeMask, kPhysicalPipeCount> relay{};
  bool has_bypass_path = false;

  static CompletionFacts AtEntry();
  void MergeFrom(const CompletionFacts& other);
  void ApplySyncEdges(const std::vector<CompletionSyncEdge>& edges);
  void ApplyWritePipes(PipeMask writes);
  void ApplySummary(const CompletionSummary& summary);
  CompletionSummary ToSummary() const;
  bool operator==(const CompletionFacts& other) const {
    return active == other.active && post_write_safe == other.post_write_safe && relay == other.relay &&
           has_bypass_path == other.has_bypass_path;
  }
  bool operator!=(const CompletionFacts& other) const { return !(*this == other); }
};

// One vertex in a sub-CFG for Kosaraju SCC completion transfer.
// Transfer executes in order: pre_syncs -> (nested_summary or write_pipes) -> post_syncs.
struct CompletionVertex {
  std::vector<CompletionSyncEdge> pre_syncs;
  PipeMask write_pipes = 0;
  std::optional<CompletionSummary> nested_summary;
  std::vector<CompletionSyncEdge> post_syncs;
};

// Instrumentation counters emitted by SolveCompletionTransfer.
struct CompletionTransferWork {
  size_t sccs = 0;
  size_t non_trivial_sccs = 0;
};

// Solved Section 8.3 completion transfer result for a sub-CFG.
struct CompletionTransferResult {
  CompletionSummary summary;
  CompletionTransferWork work;
};

// Solve Section 8.3 SCC (A, E, S, t) completion transfer from `entry` to `exit` over
// directed edges `(u -> v)` in `edges` using Kosaraju topological SCC decomposition.
CompletionTransferResult SolveCompletionTransfer(const std::vector<CompletionVertex>& vertices,
                                                 const std::vector<std::pair<size_t, size_t>>& edges, size_t entry,
                                                 size_t exit);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_COMPLETION_H_
