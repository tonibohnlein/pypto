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

#include <algorithm>
#include <iterator>
#include <memory>

#include "pypto/core/logging.h"
#include "src/ir/transforms/utils/numerical_sync_merge.h"

namespace pypto::ir::sync {
namespace {
using Ordinal = NumericalOrdinal;
using Node = NumericalNode;
using NodePtr = std::shared_ptr<const Node>;
using Row = NumericalRow;
Row EmptyForward(const Node& node) {
  Row out;
  for (size_t c = 0; c < kNumericalChains; ++c) out[c] = node.chains[c].size();
  return out;
}
Row EmptyReverse() {
  Row out;
  out.fill(-1);
  return out;
}
Row AcrossForward(const Node& node, const Row& source, const NumericalOrderQuery& query) {
  auto out = EmptyForward(*node.right);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t d = 0; d < kNumericalChains; ++d) {
      ++query.steps;
      const auto& list = node.crossing_index[c][d].forward;
      const auto found =
          std::lower_bound(list.begin(), list.end(), source[c],
                           [](const auto& entry, int64_t position) { return entry.first < position; });
      if (found != list.end()) out[d] = std::min(out[d], found->second);
    }
  }
  return out;
}
Row AcrossReverse(const Node& node, const Row& target, const NumericalOrderQuery& query) {
  auto out = EmptyReverse();
  for (size_t c = 0; c < kNumericalChains; ++c) {
    for (size_t d = 0; d < kNumericalChains; ++d) {
      ++query.steps;
      const auto& list = node.crossing_index[c][d].reverse;
      const auto after =
          std::upper_bound(list.begin(), list.end(), target[d],
                           [](int64_t position, const auto& entry) { return position < entry.first; });
      if (after != list.begin()) out[c] = std::max(out[c], std::prev(after)->second);
    }
  }
  return out;
}
Row ProjectForward(const Node& node, const Row& left, const Row& right) {
  auto out = EmptyForward(node);
  for (size_t c = 0; c < kNumericalChains; ++c) {
    const auto& l = node.left_projection[c];
    const auto& r = node.right_projection[c];
    const auto lp = std::lower_bound(l.begin(), l.end(), left[c]);
    if (lp != l.end()) {
      out[c] = lp - l.begin();
    } else {
      const auto rp = std::lower_bound(r.begin(), r.end(), right[c]);
      if (rp != r.end()) out[c] = l.size() + (rp - r.begin());
    }
  }
  return out;
}
Row ProjectReverse(const Node& node, const Row& left, const Row& right) {
  auto out = EmptyReverse();
  for (size_t c = 0; c < kNumericalChains; ++c) {
    const auto& l = node.left_projection[c];
    const auto& r = node.right_projection[c];
    const auto rp = std::upper_bound(r.begin(), r.end(), right[c]);
    if (rp != r.begin()) {
      out[c] = l.size() + (rp - r.begin()) - 1;
    } else {
      const auto lp = std::upper_bound(l.begin(), l.end(), left[c]);
      if (lp != l.begin()) out[c] = (lp - l.begin()) - 1;
    }
  }
  return out;
}
Row Reachable(const NumericalOrderQuery& query, const NodePtr& node, Ordinal event, bool completion) {
  if (!node->left) {
    auto out = EmptyForward(*node);
    for (size_t c = 0; c < kNumericalChains; ++c) {
      size_t low = 0, high = node->chains[c].size();
      while (low < high) {
        ++query.steps;
        const size_t middle = low + (high - low) / 2;
        if (query.leaf(node->leaf, event, completion, node->chains[c][middle], c % 2))
          high = middle;
        else
          low = middle + 1;
      }
      out[c] = low;
    }
    return out;
  }
  auto left = EmptyForward(*node->left), right = EmptyForward(*node->right);
  if (event < node->left->length) {
    left = Reachable(query, node->left, event, completion);
    const auto entered = AcrossForward(*node, left, query);
    for (size_t c = 0; c < kNumericalChains; ++c)
      if (entered[c] < static_cast<int64_t>(node->right->chains[c].size())) {
        for (size_t d = 0; d < kNumericalChains; ++d)
          right[d] = std::min(right[d], node->right->forward[c][entered[c]][d]);
      }
  } else {
    right = Reachable(query, node->right, event - node->left->length, completion);
  }
  return ProjectForward(*node, left, right);
}
Row Ancestors(const NumericalOrderQuery& query, const NodePtr& node, Ordinal event, bool completion) {
  if (!node->left) {
    auto out = EmptyReverse();
    for (size_t c = 0; c < kNumericalChains; ++c) {
      size_t low = 0, high = node->chains[c].size();
      while (low < high) {
        ++query.steps;
        const size_t middle = low + (high - low) / 2;
        if (query.leaf(node->leaf, node->chains[c][middle], c % 2, event, completion))
          low = middle + 1;
        else
          high = middle;
      }
      out[c] = static_cast<int64_t>(low) - 1;
    }
    return out;
  }
  auto left = EmptyReverse(), right = EmptyReverse();
  if (event >= node->left->length) {
    right = Ancestors(query, node->right, event - node->left->length, completion);
    const auto entered = AcrossReverse(*node, right, query);
    for (size_t c = 0; c < kNumericalChains; ++c)
      if (entered[c] >= 0) {
        for (size_t d = 0; d < kNumericalChains; ++d)
          left[d] = std::max(left[d], node->left->reverse[c][entered[c]][d]);
      }
  } else {
    left = Ancestors(query, node->left, event, completion);
  }
  return ProjectReverse(*node, left, right);
}
bool Query(const NumericalOrderQuery& query, const NodePtr& node, Ordinal source, bool sc, Ordinal target,
           bool tc) {
  ++query.steps;
  if (!node->left) return query.leaf(node->leaf, source, sc, target, tc);
  const auto boundary = node->left->length;
  if (source < boundary && target < boundary) return Query(query, node->left, source, sc, target, tc);
  if (source >= boundary && target >= boundary)
    return Query(query, node->right, source - boundary, sc, target - boundary, tc);
  if (source >= boundary) return false;
  const auto reachable = Reachable(query, node->left, source, sc);
  const auto entered = AcrossForward(*node, reachable, query);
  const auto ancestors = Ancestors(query, node->right, target - boundary, tc);
  for (size_t c = 0; c < kNumericalChains; ++c)
    if (entered[c] <= ancestors[c]) return true;
  return false;
}
}  // namespace
bool QueryNumericalOrder(const NumericalOrderQuery& query, const std::shared_ptr<const NumericalNode>& root,
                         NumericalOrdinal source, bool source_completion, NumericalOrdinal target,
                         bool target_completion) {
  CHECK(root && query.leaf) << "Numerical order query requires an owned root and leaf interface";
  CHECK(source >= 0 && target >= 0 && source < root->length && target < root->length)
      << "Numerical order query is outside the supplied occurrence domain";
  return Query(query, root, source, source_completion, target, target_completion);
}
}  // namespace pypto::ir::sync
