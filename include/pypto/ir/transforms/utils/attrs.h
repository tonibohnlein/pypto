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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_

#include <any>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace pypto {
namespace ir {

/// Attribute key for ``pl.pipeline(N, stage=F)`` — appears on ``ForStmt.attrs_``
/// if and only if ``ForStmt.kind_ == ForKind::Pipeline`` (bidirectional invariant
/// enforced by the structural verifier ``PipelineLoopValid``).
///
/// Lifecycle:
///   - User-written ``pl.pipeline(stage=F)``           → attr = F (any F ≥ 1)
///   - After ``LowerPipelineLoops`` (factor > 1 path)  → attr = 1 (post-lowering marker)
///   - After ``CanonicalizeIOOrder``                   → attr stripped, kind demoted
///
/// ``LowerPipelineLoops`` triggers on attr > 1; attr == 1 is a no-op trigger
/// (loop is left intact for ``CanonicalizeIOOrder`` to reorder and demote).
inline constexpr const char* kPipelineStagesAttr = "pipeline_stages";

/// Optional ``bool`` policy attr on a ``ForKind::Pipeline`` ``ForStmt``: when
/// ``false``, ``CanonicalizeIOOrder`` keeps store-like ops in the *compute*
/// stage tier instead of floating them to the bottom ``Store`` tier.
///
/// Rationale: the default (absent ⇒ ``true``) floats all sibling-iteration
/// stores below all compute, which keeps both iterations' *output* tiles
/// co-live — a ping-pong on the output buffer. For the full-K M/N matmul
/// pipeline each iteration writes a *different, large* L0C result, so output
/// ping-pong would force two L0C buffers co-live (``2·m·n·bytes_c``) while the
/// tile chooser budgets only one (``double_buffer_c == false``) — an L0C
/// overflow at allocation. Setting this ``false`` yields the one-accumulator
/// schedule ``extract_i, extract_{i+1}, matmul_i, store_i, matmul_{i+1}, …``:
/// the moving-operand extract is still double-buffered (Load tier, hoisted),
/// but ``store_i`` drains before ``matmul_{i+1}`` overwrites the single L0C
/// accumulator. Consumed (stripped) by ``CanonicalizeIOOrder`` alongside
/// ``pipeline_stages``.
inline constexpr const char* kPipelineOverlapStoresAttr = "pipeline_overlap_stores";

/// Return a copy of `attrs` with any entry matching `key` removed. The order of
/// the remaining entries is preserved.
inline std::vector<std::pair<std::string, std::any>> StripAttr(
    const std::vector<std::pair<std::string, std::any>>& attrs, std::string_view key) {
  std::vector<std::pair<std::string, std::any>> out;
  out.reserve(attrs.size());
  for (const auto& [k, v] : attrs) {
    if (k == key) continue;
    out.emplace_back(k, v);
  }
  return out;
}

}  // namespace ir
}  // namespace pypto

#endif  // PYPTO_IR_TRANSFORMS_UTILS_ATTRS_H_
