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

#include <limits>
#include <map>
#include <set>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
std::optional<int64_t> EvaluateSyncCoordinate(const SyncCoordinateProgram& program, size_t root,
                                              const std::map<const ForStmt*, int64_t>& original_values,
                                              const std::map<size_t, int64_t>& parameters,
                                              std::optional<int64_t> block) {
  using Kind = SyncCoordinateNode::Kind;
  if (root >= program.nodes.size()) return std::nullopt;
  std::set<size_t> needed;
  std::vector<size_t> pending{root};
  while (!pending.empty()) {
    const auto id = pending.back();
    pending.pop_back();
    if (!needed.insert(id).second) continue;
    const auto& n = program.nodes[id];
    auto dependency = [&](size_t child) {
      if (child >= id) return false;
      pending.push_back(child);
      return true;
    };
    if (n.kind == Kind::Scale || n.kind == Kind::ModularShift || (n.kind == Kind::Loop && n.dynamic_start)) {
      if (!dependency(n.left)) return std::nullopt;
    } else if (n.kind == Kind::Add || n.kind == Kind::FloorDivide || n.kind == Kind::Remainder) {
      if (!dependency(n.left) || !dependency(n.right)) return std::nullopt;
    }
  }
  std::map<size_t, int64_t> values;
  for (auto id : needed) {
    const auto& n = program.nodes[id];
    __int128 value = 0;
    switch (n.kind) {
      case Kind::Constant:
        value = n.upper;
        break;
      case Kind::Block:
        if (!block) return std::nullopt;
        value = *block;
        break;
      case Kind::Parameter: {
        const auto found = parameters.find(n.parameter);
        if (found == parameters.end()) return std::nullopt;
        value = found->second;
        break;
      }
      case Kind::Loop: {
        const auto found = original_values.find(n.frame.get());
        if (found == original_values.end() || n.step <= 0) return std::nullopt;
        value = found->second;
        const auto start = n.dynamic_start ? values.at(n.left) : n.start;
        if (value < start || (value - start) % n.step) return std::nullopt;
        break;
      }
      case Kind::Scale:
        value = __int128(values.at(n.left)) * n.factor;
        break;
      case Kind::ModularShift:
        value = __int128(values.at(n.left)) - n.factor;
        break;
      case Kind::Add:
        value = __int128(values.at(n.left)) + values.at(n.right);
        break;
      case Kind::FloorDivide:
      case Kind::Remainder:
        if (values.at(n.right) <= 0) return std::nullopt;
        value = n.kind == Kind::FloorDivide ? values.at(n.left) / values.at(n.right)
                                            : values.at(n.left) % values.at(n.right);
        break;
    }
    if (value < n.lower || value > n.upper || value > std::numeric_limits<int64_t>::max())
      return std::nullopt;
    values.emplace(id, static_cast<int64_t>(value));
  }
  return values.at(root);
}
}  // namespace pypto::ir::sync
