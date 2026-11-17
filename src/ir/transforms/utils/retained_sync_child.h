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
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"

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
  void ValidateFor(const FunctionPtr& owner, const std::string& target) const;

 private:
  friend NativeSyncChildExport ExportNativeSyncChild(const CompactSyncResult&, const ForStmtPtr&,
                                                     const std::string&);
  NativeSyncChild(CompactSyncResult mathematics, ForStmtPtr scope, std::string target)
      : mathematics_(std::move(mathematics)), scope_(std::move(scope)), target_(std::move(target)) {}
  CompactSyncResult mathematics_;
  ForStmtPtr scope_;
  std::string target_;
};

// Routing hint only. LowerSync independently validates the marked token region,
// lexical descriptor bindings, entry acknowledgments and directional resources.
inline constexpr const char* kClosedSyncChildAttr = "pypto.closed_sync_child";
inline constexpr const char* kClosedSyncChildExternalPolicyAttr =
    "pypto.closed_sync_child_external_policy";
inline constexpr const char* kClosedSyncChildFirstVisitAttr =
    "pypto.closed_sync_child_first_visit";

// Returns true when `loop` is a sequential ForStmt whose symbolic non-negative
// SPMD start and constant bounds (`0 < stop <= step` with `step > 1`) ensure
// at most one dynamic iteration (0 or 1 trip, zero backedges).
bool IsSpmdAtMostOneTripFor(const ForStmtPtr& loop,
                             const std::unordered_map<const Var*, ExprPtr>& ssa_defs);

// Inspect actual selected instruction effects, including implicit effects.
// Returns true only when external effects exist and every such effect reads.
struct SyncChildEffectWork {
  size_t statements = 0, expressions = 0, native_qualifications = 0;
};
bool HasReadonlyExternalSyncEffects(const ForStmtPtr& scope, FunctionType core, const std::string& target,
                                    SyncChildEffectWork* work = nullptr,
                                    bool allow_disjoint_gm_writes = false);
SyncParameterBounds ProjectSyncParameterBounds(const FunctionPtr& owner, const FunctionPtr& binding_owner,
                                               const SyncParameterBounds& owner_bounds);

struct SyncRegionIndexWork {
  size_t source_nodes;
  size_t parameter_nodes = 0;
};
struct SyncRegionBindWork {
  size_t scope_nodes = 0;
  size_t expression_nodes = 0;
  size_t binding_definitions = 0;
  size_t ancestor_links = 0;
  size_t logical_index_nodes = 0;
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
  SyncRegionContext Bind(const ForStmtPtr& scope,
                         SyncExternalPolicy external_policy = SyncExternalPolicy::Geometry,
                         SyncRegionBindWork* work = nullptr) const;
  size_t SourceNodes() const;
  size_t ParameterNodes() const;

 private:
  class State;
  std::shared_ptr<State> state_;
};

struct EventfulSyncChildAttempt;
class EventfulSyncChild {
 public:
  const FunctionPtr& Owner() const { return context_.original_owner; }
  const ForStmtPtr& Scope() const { return context_.original_scope; }
  const StmtPtr& LogicalRecipe() const { return logical_recipe_; }
  const std::vector<SyncDirection>& RearmDirections() const { return rearm_dirs_; }
  std::map<std::string, size_t> Describe() const;
  void ValidateFor(const FunctionPtr& owner, const std::string& target) const;

 private:
  friend struct RetainedSyncChildDiagnostics;
  SyncRegionContext context_;
  CompactSyncResult mathematics_;
  StmtPtr logical_recipe_;
  std::string target_;
  std::vector<SyncDirection> rearm_dirs_;
  EventfulSyncChild(SyncRegionContext context, CompactSyncResult mathematics, StmtPtr logical_recipe,
                    std::string target, std::vector<SyncDirection> rearm_dirs)
      : context_(std::move(context)),
        mathematics_(std::move(mathematics)),
        logical_recipe_(std::move(logical_recipe)),
        target_(std::move(target)),
        rearm_dirs_(std::move(rearm_dirs)) {}
  friend std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr&,
                                                                           const ForStmtPtr&,
                                                                           const std::string&,
                                                                           const SyncRegionBindings&);
  friend EventfulSyncChildAttempt TryPrepareEventfulSyncChild(
      const FunctionPtr&, const ForStmtPtr&, const std::string&, const SyncRegionBindings&,
      const SyncParameterBounds&, const std::optional<std::pair<int64_t, int64_t>>&);
};
std::shared_ptr<const EventfulSyncChild> PrepareEventfulSyncChild(const FunctionPtr& owner,
                                                                  const ForStmtPtr& scope,
                                                                  const std::string& target,
                                                                  const SyncRegionBindings& bindings);

enum class EventfulSyncChildStage { Effects, Binding, Analysis, Qualification, ExecutableExport, Ready };
const char* EventfulSyncChildStageName(EventfulSyncChildStage stage);
// An export refusal retains its successful immutable mathematics. These are
// provider-owned attempt results, not a change to the exact analysis guarantee.
struct EventfulSyncChildAttempt {
  ForStmtPtr scope;
  EventfulSyncChildStage stage = EventfulSyncChildStage::Effects;
  std::string reason;
  SyncRegionBindWork binding_work;
  SyncChildEffectWork effect_work;
  // Descriptive inclusive phase times also account for refused attempts.
  std::map<EventfulSyncChildStage, double> seconds;
  size_t logical_record_sites = 0, logical_wait_sites = 0;
  std::shared_ptr<const CompactSyncResult> mathematics;
  std::shared_ptr<const EventfulSyncChild> child;
};
EventfulSyncChildAttempt TryPrepareEventfulSyncChild(
    const FunctionPtr& owner, const ForStmtPtr& scope, const std::string& target,
    const SyncRegionBindings& bindings, const SyncParameterBounds& parameters = {},
    const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt);
struct AutomaticSyncChildren {
  std::vector<EventfulSyncChildAttempt> attempts;
  size_t discovery_nodes = 0;
  size_t index_nodes = 0;
  size_t parameter_nodes = 0;
};
// One discovery and one binding index per owner; candidates are disjoint whole
// sequential For invocations without nested For/While scopes. Costs additionally
// include each provider attempt and the requested binding/interface output.
AutomaticSyncChildren PrepareAutomaticSyncChildren(
    const FunctionPtr& owner, const std::string& target, const SyncParameterBounds& parameters = {},
    const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt);

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
