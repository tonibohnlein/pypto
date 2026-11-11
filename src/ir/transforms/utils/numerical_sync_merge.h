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

#ifndef SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_MERGE_H_
#define SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_MERGE_H_
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
namespace pypto::ir::sync {
using NumericalOrdinal = __int128;
constexpr size_t kNumericalChains = 2 * kPipeCount;
using NumericalEvent = std::pair<NumericalOrdinal, bool>;
using NumericalPosition = std::pair<size_t, size_t>;
using NumericalRow = std::array<int64_t, kNumericalChains>;
struct NumericalBoundary {
  std::optional<NumericalOrdinal> first_writer, last_writer;
  std::array<std::optional<NumericalOrdinal>, kPipeCount> first_readers{}, last_readers{};
};
struct NumericalCell {
  size_t storage;
  int64_t slot, row, column;
};
struct NumericalCrossing {
  NumericalPosition source, target;
  bool native;
};
struct NumericalCrossIndex {
  // source position -> suffix minimum target; target -> prefix maximum source.
  std::vector<std::pair<int64_t, int64_t>> forward, reverse;
};
struct NumericalNode {
  NumericalOrdinal length = 0;
  size_t leaf = 0;
  std::shared_ptr<const NumericalNode> left, right;
  std::vector<NumericalBoundary> storage;
  // Original native site positions; populated only on leaves for predecessor
  // queries. Shared repeat nodes do not duplicate or enumerate their visits.
  std::array<std::vector<size_t>, kPipeCount> native_sites;
  // Existing Mmad footprint protection at native M-chain boundary events.
  // The target must be a continuation; an initialization cannot inherit it.
  std::optional<size_t> first_m_continuation, last_m_qualified;
  std::array<std::optional<NumericalOrdinal>, kPipeCount> first{}, last{};
  std::array<std::vector<NumericalOrdinal>, kNumericalChains> chains;
  std::array<std::vector<NumericalRow>, kNumericalChains> forward, reverse;
  std::map<NumericalEvent, NumericalPosition> positions;
  std::vector<NumericalCrossing> crossings;
  std::array<std::array<NumericalCrossIndex, kNumericalChains>, kNumericalChains> crossing_index;
  std::array<std::vector<int64_t>, kNumericalChains> left_projection, right_projection;
};
// Source identities use this unshared layout, not the merge DAG. Summary
// doubling may reuse a node, but each source leaf has one occurrence path.
struct NumericalSourceLayout {
  std::optional<size_t> leaf;
  ForStmtPtr frame;
  int64_t count = 1;
  NumericalOrdinal body_length = 0, length = 0;
  // Existing summary objects used by one source body and its adjacent-visit
  // seam. Export traverses this source tree, never every use of the repeat DAG.
  std::shared_ptr<const NumericalNode> body, summary, seam;
  std::vector<NumericalSourceLayout> children;
  std::vector<NumericalOrdinal> offsets;
};
struct NumericalOccurrencePath {
  NumericalOrdinal first = 0;
  // Outermost first: radix and reference-order payload stride.
  std::vector<std::pair<int64_t, NumericalOrdinal>> frames;
};
// Complete schedule proof for a byte domain: full monotone row fills followed
// by finite interval readers covering every row. Witnesses suffice for crossing
// generation; they are NOT a materialized cell partition or a selector answer.
struct CollectiveSyncDomain {
  int64_t row_bytes = 0, rows = 0;
  std::vector<size_t> fill_leaves;
  std::vector<int64_t> witness_columns;
};
class NumericalSyncHierarchy {
 public:
  std::shared_ptr<const NumericalNode> root;
  std::vector<NumericalCell> cells;
  std::map<size_t, CollectiveSyncDomain> collective_domains;
  std::vector<NumericalOrdinal> segment_offsets;
  std::optional<NumericalSourceLayout> source_layout;
  std::vector<NumericalOccurrencePath> source_paths;
};
NumericalOrdinal NumericalAdd(NumericalOrdinal a, NumericalOrdinal b);
NumericalBoundary MergeNumericalBoundary(const NumericalBoundary& a, const NumericalBoundary& b,
                                         NumericalOrdinal offset);
std::set<NumericalEvent> NumericalPorts(const NumericalNode& node, size_t& visits);
void IndexNumericalPositions(NumericalNode& node);
void InvertNumericalRows(NumericalNode& node);
void CountNumericalInterface(const NumericalNode& node, CompactSyncResult& result);
// Order-only service: supplied child order, complete parent ports and explicit
// crossing edges. It performs no access recovery, storage bridging or reduction.
// Children have sorted original-event chains and exact monotone forward/reverse
// rows. Supplied edges include native chain links across the boundary; parent
// ports retain every event needed by later joins. The parent must be fresh with
// left/right/length initialized. The indexed row kernel costs O(k P + k^2 E);
// map/set indexing adds O(P log P), separately from interface construction.
// P counts all inspected child and parent ports, including discarded ports.
void BuildNumericalMergeOrder(const std::shared_ptr<NumericalNode>& node,
                              const std::set<NumericalEvent>& ports,
                              std::vector<NumericalCrossing> crossings);
// Leaf answers are reflexive event reachability and monotone on each supplied
// native start/completion chain. They must describe the same order as the leaf
// rows. The callback and work counter are borrowed for this synchronous query.
struct NumericalOrderQuery {
  std::function<bool(size_t, NumericalOrdinal, bool, NumericalOrdinal, bool)> leaf;
  size_t& steps;
};
bool QueryNumericalOrder(const NumericalOrderQuery& query, const std::shared_ptr<const NumericalNode>& root,
                         NumericalOrdinal source, bool source_completion, NumericalOrdinal target,
                         bool target_completion);
std::shared_ptr<const NumericalNode> MergeNumericalNodes(std::shared_ptr<const NumericalNode> a,
                                                         std::shared_ptr<const NumericalNode> b,
                                                         CompactSyncResult& result);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_MERGE_H_
