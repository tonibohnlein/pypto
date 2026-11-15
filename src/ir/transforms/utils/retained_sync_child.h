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
#include <map>
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

// Routing hint only. LowerSync independently validates the marked token region,
// lexical descriptor bindings, entry acknowledgments and directional resources.
inline constexpr const char* kClosedSyncChildAttr = "pypto.closed_sync_child";

struct SyncRegionIndexWork {
  size_t source_nodes;
};
struct SyncRegionContext {
  FunctionPtr original_owner;
  ForStmtPtr original_scope;
  std::vector<StmtPtr> ancestors;
  FunctionPtr binding_owner;
  // Shared build cost is counted once for the request batch, not per child.
  std::shared_ptr<const SyncRegionIndexWork> index_work;
  size_t query_scope_nodes = 0, binding_definitions = 0;
};
// One original-owner index per provider batch or lowering request. Queries
// walk only the requested original scope, ancestors and necessary bindings.
// No trip counts are unfolded and no process-global context cache is used.
class SyncRegionBindings {
 public:
  SyncRegionBindings(FunctionPtr owner, std::string target);
  SyncRegionContext Bind(const ForStmtPtr& scope) const;

 private:
  class State;
  std::shared_ptr<State> state_;
};

class EventfulSyncChild {
 public:
  const FunctionPtr& Owner() const { return context_.original_owner; }
  const ForStmtPtr& Scope() const { return context_.original_scope; }
  const StmtPtr& LogicalRecipe() const { return logical_recipe_; }
  const std::vector<PipeType>& Pipes() const { return pipes_; }
  std::map<std::string, size_t> Describe() const;
  void ValidateFor(const FunctionPtr& owner, const std::string& target) const;

 private:
  SyncRegionContext context_;
  CompactSyncResult mathematics_;
  StmtPtr logical_recipe_;
  std::string target_;
  std::vector<PipeType> pipes_;
  EventfulSyncChild(SyncRegionContext context, CompactSyncResult mathematics, StmtPtr logical_recipe,
                    std::string target, std::vector<PipeType> pipes)
      : context_(std::move(context)),
        mathematics_(std::move(mathematics)),
        logical_recipe_(std::move(logical_recipe)),
        target_(std::move(target)),
        pipes_(std::move(pipes)) {}
  friend std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr&,
                                                                           const ForStmtPtr&,
                                                                           const std::string&,
                                                                           const SyncRegionBindings&);
};
std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr& owner,
                                                                  const ForStmtPtr& scope,
                                                                  const std::string& target,
                                                                  const SyncRegionBindings& bindings);

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
