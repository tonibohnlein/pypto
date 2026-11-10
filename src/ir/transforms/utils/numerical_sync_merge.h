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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_MERGE_H_
#define PYPTO_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_MERGE_H_
#include <array>
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
  std::array<std::optional<NumericalOrdinal>, kPipeCount> first{}, last{};
  std::array<std::vector<NumericalOrdinal>, kNumericalChains> chains;
  std::array<std::vector<NumericalRow>, kNumericalChains> forward, reverse;
  std::map<NumericalEvent, NumericalPosition> positions;
  std::vector<NumericalCrossing> crossings;
  std::array<std::array<NumericalCrossIndex, kNumericalChains>, kNumericalChains> crossing_index;
  std::array<std::vector<int64_t>, kNumericalChains> left_projection, right_projection;
};
class NumericalSyncHierarchy {
 public:
  std::shared_ptr<const NumericalNode> root;
  std::vector<NumericalCell> cells;
  std::vector<NumericalOrdinal> segment_offsets;
};
NumericalOrdinal NumericalAdd(NumericalOrdinal a, NumericalOrdinal b);
NumericalBoundary MergeNumericalBoundary(const NumericalBoundary& a, const NumericalBoundary& b,
                                         NumericalOrdinal offset);
std::set<NumericalEvent> NumericalPorts(const NumericalNode& node, size_t& visits);
void IndexNumericalPositions(NumericalNode& node);
void InvertNumericalRows(NumericalNode& node);
void CountNumericalInterface(const NumericalNode& node, CompactSyncResult& result);
std::shared_ptr<const NumericalNode> MergeNumericalNodes(std::shared_ptr<const NumericalNode> a,
                                                         std::shared_ptr<const NumericalNode> b,
                                                         CompactSyncResult& result);
}  // namespace pypto::ir::sync
#endif
