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

#ifndef SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_BUILDER_H_
#define SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_BUILDER_H_

#include <memory>
#include <utility>

#include "pypto/core/error.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Internal construction is move-only and cannot cache/export scoped demands.
// Publish consumes its only mutable state handle; callers must also relinquish
// local mutable graph/hierarchy handles before the published result escapes.
class CompactSyncBuilder final : public CompactSyncView {
 public:
  CompactSyncBuilder() : CompactSyncBuilder(std::make_shared<CompactSyncState>()) {}
  CompactSyncBuilder(const CompactSyncBuilder&) = delete;
  CompactSyncBuilder& operator=(const CompactSyncBuilder&) = delete;
  CompactSyncBuilder(CompactSyncBuilder&&) = default;
  CompactSyncBuilder& operator=(CompactSyncBuilder&&) = default;
  using CompactSyncView::State;
  CompactSyncState& State() {
    INTERNAL_CHECK(mutable_state_) << "Cannot modify a published synchronization result";
    return *mutable_state_;
  }
  CompactSyncResult Publish() && {
    INTERNAL_CHECK(mutable_state_) << "Synchronization result already published";
    mutable_state_.reset();
    return CompactSyncResult(std::move(state_), std::move(diagnostics_));
  }

 private:
  explicit CompactSyncBuilder(std::shared_ptr<CompactSyncState> state)
      : CompactSyncView(state, std::make_shared<CompactSyncDiagnostics>()),
        mutable_state_(std::move(state)) {}
  std::shared_ptr<CompactSyncState> mutable_state_;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_COMPACT_SYNC_BUILDER_H_
