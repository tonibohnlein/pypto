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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_FINITE_SYNC_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_FINITE_SYNC_ANALYSIS_H_
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/pipe.h"

namespace pypto::ir::sync {
constexpr size_t kPipeCount = static_cast<size_t>(PipeType::ALL);
using Cell = uint64_t;
using Demand = std::pair<size_t, size_t>;
/// Columns [0,k) are maximum start-ancestor ranks; [k,2k) are
/// completion-ancestor ranks. Ranks are one-based among executed occurrences
/// on that pipe; zero is empty. Rows include the target event itself, while
/// Precedes excludes reflexive pairs. No source statement numbering is used
/// as a substitute for an executed rank.
using RankRow = std::array<size_t, 2 * kPipeCount>;

/// Resolved finite occurrence. Indices retain original identity including
/// absent occurrences. Cells denote physical classes, never SSA identities.
struct FiniteOccurrence {
  PipeType pipe;
  bool present;
  std::vector<Cell> reads;
  std::vector<Cell> writes;
};
struct FiniteSyncInput {
  std::vector<FiniteOccurrence> occurrences;
  /// Provider-certified complete domain: omitted cells are unavailable.
  std::vector<Cell> covered_cells;
  /// True certifies exact physical conflicts; false denotes sound enclosures.
  /// This is a provider premise, not an access-recovery proof by the reducer.
  bool exact_accesses = false;
  /// Forward C->I edges apply when both original endpoints occur.
  std::vector<Demand> prerequisites;
  /// Qualified native C->I edges, e.g. scalar serialization. No hardware
  /// exemption is inferred from a pipe name in this numerical kernel.
  std::vector<Demand> native_order;
  /// Provider-certified same-pipe writer-chain protection, NOT a native C->I
  /// edge or an arbitrary edge-removal predicate. The certificate must preserve
  /// frontier substitution across protected writers. Pure reader hazards and
  /// explicit prerequisites stay mandatory. Callback work is charged separately.
  std::function<bool(size_t, size_t, Cell)> protects_storage;
};
struct StorageBoundary {
  std::optional<size_t> first_writer;
  std::optional<size_t> last_writer;
  std::array<std::optional<size_t>, kPipeCount> first_readers{};
  std::array<std::optional<size_t>, kPipeCount> last_readers{};
};
struct FiniteSyncWork {
  size_t access_entries = 0;
  size_t cell_visits = 0;
  size_t reader_visits = 0;
  size_t witnesses = 0;
  size_t candidate_tests = 0;
  size_t demand_joins = 0;
};
/// Numerical result; does not assert executable endpoints or ID feasibility.
struct FiniteSyncResult {
  bool exact_accesses = false;
  std::vector<PipeType> pipes;
  std::vector<size_t> ranks;  // Zero means proved absent.
  std::vector<Demand> generators;
  std::vector<Demand> demands;
  std::vector<RankRow> starts;
  std::vector<RankRow> completions;
  std::unordered_map<Cell, StorageBoundary> storage;
  FiniteSyncWork work;
  /// Strict reachability for all four start/completion combinations.
  bool Precedes(size_t source, bool source_completion, size_t target, bool target_completion) const;
  /// Nullopt means unavailable; an empty boundary proves no access.
  std::optional<StorageBoundary> Boundary(Cell cell) const;
};
/// Expected O(hk+n+c+e+k*(n+a+f)) work for h covered cells, n occurrences,
/// c access incidences, e prerequisites, a native edges and f retained demands.
/// With k fixed hardware pipes this is linear in input and output. No
/// read/read pair scan, all-pairs closure, or symbolic guard reconstruction.
FiniteSyncResult AnalyzeFiniteSync(const FiniteSyncInput& input);
}  // namespace pypto::ir::sync
#endif  // PYPTO_IR_TRANSFORMS_UTILS_FINITE_SYNC_ANALYSIS_H_
