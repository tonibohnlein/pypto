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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_BOUNDARY_GRAPH_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_BOUNDARY_GRAPH_H_
#include <map>
#include <optional>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Existing explicit port graph, shared by the legacy mathematical reducer and
// structured-token interpretation. Contains only supplied child order, native
// chain boundaries and explicitly added crossings. No access analysis.
// O(sum P_child^2) construction and O(P+E) graph search; child query costs are
// separate. Ports are first/final visits, never an unfolding of trip counts.
class SyncBoundaryGraph {
 public:
  using Event = std::pair<Occurrence, bool>;
  explicit SyncBoundaryGraph(const std::vector<CompactSegmentResult>& segments);
  size_t Node(const Event& event);
  void Add(const Event& source, const Event& target);
  bool Path(size_t source, size_t target, std::optional<std::pair<size_t, size_t>> skip = std::nullopt) const;
  bool Precedes(const std::vector<CompactSegmentResult>& segments, const Occurrence& a, bool sc,
                const Occurrence& b, bool tc) const;
  size_t Edges() const;
  size_t pair_tests = 0;

 private:
  std::map<Event, size_t> nodes_;
  std::vector<std::vector<std::pair<Event, size_t>>> ports_;
  std::vector<std::vector<size_t>> graph_;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_BOUNDARY_GRAPH_H_
