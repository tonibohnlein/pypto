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

#ifndef SRC_IR_TRANSFORMS_UTILS_RETAINED_SYNC_CHILD_H_
#define SRC_IR_TRANSFORMS_UTILS_RETAINED_SYNC_CHILD_H_

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
class NativeSyncChild;
struct NativeSyncChildExport;
NativeSyncChildExport ExportNativeSyncChild(const CompactSyncResult& mathematics, const ForStmtPtr& scope,
                                            const std::string& target);

// Provider-owned native-only child. The existing retained numerical summary
// proves no active nonnative internal demands; all original native operations
// are qualified and have no flag claims. Consequently the ORIGINAL payload
// subtree is the exact internal recipe, with empty notification entry/exit state.
// This establishes neither payload completion at exit nor exact GM selectors.
// The first export accepts only local data effects: a supplied disjoint-GM
// premise is not inherited by a conservative may-alias parent. Caller/launch
// range-specialized analyses also remain unavailable until the parent carries
// and validates those same supplied premises.
//
// One request scans the retained source/operation inventory and the selected
// summary DAG; R independent requests currently cost O(R*N) context lookup,
// plus visited summary/interface nodes. No trip-count unfolding is performed.
//
// Context cannot be transplanted: original owner, scope, target, placement,
// alias model and caller premises remain those of the immutable mathematics.
// Eventful children need matching confinement, palette and rearm exports, which
// this type deliberately cannot represent. No detached child is reanalyzed.
class NativeSyncChild {
 public:
  const FunctionPtr& Owner() const { return mathematics_.State().input.owner; }
  const ForStmtPtr& Scope() const { return scope_; }
  const CompactSyncResult& Mathematics() const { return mathematics_; }
  const std::vector<backend::BufferSyncOperation>& Operations() const { return operations_; }
  void ValidateFor(const FunctionPtr& owner, const std::string& target) const;

 private:
  friend NativeSyncChildExport ExportNativeSyncChild(const CompactSyncResult&, const ForStmtPtr&,
                                                     const std::string&);
  NativeSyncChild(CompactSyncResult mathematics, ForStmtPtr scope, std::string target,
                  std::vector<backend::BufferSyncOperation> operations)
      : mathematics_(std::move(mathematics)),
        scope_(std::move(scope)),
        target_(std::move(target)),
        operations_(std::move(operations)) {}
  CompactSyncResult mathematics_;
  ForStmtPtr scope_;
  std::string target_;
  std::vector<backend::BufferSyncOperation> operations_;
};

enum class NativeSyncChildStatus {
  Ready,
  ForeignContext,
  MissingRetainedScope,
  UnavailableSummary,
  NonNativeDemands,
  UnqualifiedOperation,
  NativeResources,
  AccessPremises
};
const char* NativeSyncChildStatusName(NativeSyncChildStatus status);
struct NativeSyncChildExport {
  NativeSyncChildStatus status = NativeSyncChildStatus::MissingRetainedScope;
  std::string reason;
  std::shared_ptr<const NativeSyncChild> child;
  size_t source_nodes = 0, summary_nodes = 0, demands = 0, operations = 0;
};
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_RETAINED_SYNC_CHILD_H_
