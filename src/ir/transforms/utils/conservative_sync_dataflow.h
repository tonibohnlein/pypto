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

#ifndef SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_DATAFLOW_H_
#define SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_DATAFLOW_H_

#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "src/ir/transforms/utils/conservative_sync_completion.h"
#include "src/ir/transforms/utils/retained_sync_child.h"

namespace pypto::ir::sync {

// Select an inclusion-minimal ordered subset of advertised completion rows.
// The producer owns the meaning and compatibility of rows/requirements. This
// arithmetic kernel proves only coverage under that supplied immutable matrix.
// O(rows*requirements), with no subset search or minimum-cardinality claim.
inline std::vector<size_t> SelectConservativeCompletionCover(
    const std::vector<bool>& base, const std::vector<std::vector<bool>>& advertised) {
  std::vector<size_t> counts(base.size());
  for (const auto& row : advertised) {
    INTERNAL_CHECK(row.size() == base.size()) << "Internal error: completion advertisement arity mismatch";
    for (size_t requirement = 0; requirement < base.size(); ++requirement)
      counts[requirement] += row[requirement];
  }
  for (size_t requirement = 0; requirement < base.size(); ++requirement)
    INTERNAL_CHECK(base[requirement] || counts[requirement] > 0)
        << "Internal error: completion advertisements do not cover a requested prerequisite";
  std::vector<size_t> selected;
  for (size_t candidate = 0; candidate < advertised.size(); ++candidate) {
    const auto& row = advertised[candidate];
    bool removable = true;
    for (size_t requirement = 0; requirement < base.size(); ++requirement) {
      if (!base[requirement] && row[requirement] && counts[requirement] == 1) {
        removable = false;
        break;
      }
    }
    if (removable) {
      for (size_t requirement = 0; requirement < base.size(); ++requirement)
        if (row[requirement]) --counts[requirement];
    } else {
      selected.push_back(candidate);
    }
  }
  return selected;
}

// A provider-owned conservative storage class, never an exact selector.
// Domain zero is a wildcard over its whole memory space. Nonzero IDs name this
// immutable analysis's catalog only; they are not global storage identities.
// write=false denotes a possible read, write=true a possible write.
using ConservativeAccessFact = std::tuple<MemorySpace, PipeType, bool, size_t>;

struct ConservativeStorageDomain {
  size_t id;
  MemorySpace space;
  std::optional<backend::BufferSyncByteWindow> enclosure;
  ExprPtr original_root;  // Representative of a proven GM alias class.
};
using ConservativeAccessSet = std::set<ConservativeAccessFact>;

// A selected transport adapter can supply checked local effects for the
// complete original owner. A call without a pipe has no asynchronous issue
// at this cut. Actual transfer and ready/credit commands remain in the IR.
struct ImportedLocalEffect {
  MemorySpace space;
  bool write;
  std::optional<backend::BufferSyncByteWindow> enclosure;
  std::optional<size_t> gm_formal = std::nullopt;
};
struct ImportedLocalCut {
  const Stmt* statement;
  std::optional<PipeType> pipe;
  std::vector<ImportedLocalEffect> effects;
  bool has_native_claims = false;
  // The reported pipe describes only the data effect of a transport call.
  // Its native ready/credit, cursor and queue operations are opaque here.
  bool is_transport = false;

  bool HasOpaqueProtocol() const { return is_transport || has_native_claims; }
};
using ImportedLocalCuts = std::unordered_map<const Call*, ImportedLocalCut>;
struct ImportedLocalSync {
  FunctionPtr owner;
  ImportedLocalCuts calls;
  const IfStmt* terminal_branch = nullptr;
  // Populated only after every invocation has certified the GM partition.
  std::map<size_t, ExprPtr> gm_domains{};
};

// An allocated SET may move before these producer effects. Its WAIT and all
// other services stay at their original cuts. A latch move applies only to
// the steady visits of the original loop identified by the induction variable.
struct EarlierPeerPrefix {
  const Stmt* consumer;
  PipeType source;
  std::vector<const Stmt*> displaced;
  const Var* backedge_iv = nullptr;
  // A same-sequence reader release captures immediately after this payload.
  // Its displaced list names every later payload up to the consumer in
  // reverse order, including payloads on other queues.
  const Stmt* reader_record = nullptr;
};

struct EarlierReaderRelease {
  const Stmt* record_after;
  EarlierPeerPrefix delivery;
};

struct PhysicalReaderReleasePlan {
  FunctionPtr function;
  std::vector<EarlierPeerPrefix> moves;
};

struct ConservativeLoopEntryServices {
  ForStmtPtr loop;
  bool guard_nonempty;
  std::vector<PipeType> entry_only;
  std::vector<PipeType> backedge_only;
  // A steady service can be unguarded when its first publication is empty.
  std::vector<PipeType> empty_entry_sources;
};

struct ConservativePrefixCapture {
  StmtPtr consumer;
  PipeType destination;
};

struct ConservativeGuardedCapture {
  ExprPtr condition;
  PipeType destination;
};

struct ConservativeSyncSite {
  StmtPtr cut;  // Original statement, immediately before its payload call.
  CallPtr operation;
  PipeType pipe;
  std::optional<backend::BufferSyncOperation> buffer_operation;
  ConservativeAccessSet effects;
  ConservativeAccessSet incoming;  // Unavailable unless HasIncomingDiagnostics() is true.
  // Full-overwrite (space, domain) pairs proven to overwrite the entire domain cleanly.
  std::vector<std::pair<MemorySpace, size_t>> refreshed_domains;
  // Baseline storage obligations and the ordered selected subset. Non-storage
  // child/rearming services are protected separately and never filtered here.
  std::vector<PipeType> source_pipes;
  std::vector<PipeType> required_source_pipes;
  // A separately certified producer-side service, after this payload. The
  // destination pipe has no later payload before the next loop visit.
  std::optional<PipeType> post_destination;
  // Phase-only subsets of source_pipes. All other services run on both paths.
  std::optional<ConservativeLoopEntryServices> loop_entry;
  // Record before this payload and wait before the following independent
  // consumer in the same straight-line statement sequence.
  std::optional<ConservativePrefixCapture> precise_capture;
  // A first-visit handoff whose guard guarantees that the receiver executes.
  std::optional<ConservativeGuardedCapture> guarded_capture;
};

struct ConservativeSyncChildEntry {
  ForStmtPtr scope;
  ConservativeAccessSet incoming, effects;  // incoming requires HasIncomingDiagnostics().
  // A child has several participating pipes, never one fabricated payload pipe.
  std::vector<std::pair<PipeType, PipeType>> services;
  std::shared_ptr<const EventfulSyncChild> eventful;
  std::vector<std::pair<PipeType, PipeType>> required_services;
  CompletionSummary completion = CompletionSummary::Identity();
  bool first_visit = false;
  SeqStmtsPtr window_seq;
  size_t window_begin = 0;
  size_t window_end = 0;
  std::optional<InlineExactSubBlock> inline_subblock;
  bool IsSubblockSlice() const { return static_cast<bool>(window_seq); }
};

struct ConservativeSyncCost {
  size_t statements = 0;
  size_t expression_visits = 0;
  size_t dataflow_visits = 0;
  size_t retained_facts = 0;
  size_t services = 0;
  size_t baseline_services = 0;
  size_t suppressed_services = 0;
  size_t certificate_visits = 0;
  size_t certificate_omissions = 0;
  size_t killed_facts = 0;
  size_t parent_links = 0;
  size_t enclosure_lookups = 0;
  size_t storage_domains = 0;
  size_t conflict_summary_facts = 0;
  size_t conflict_queries = 0;
  size_t incoming_diagnostic_facts = 0;
  size_t control_snapshot_facts = 0;
  size_t control_merge_facts = 0;
  size_t range_indexed_definitions = 0;
  size_t range_footprint_queries = 0;
  size_t range_fallback_definitions = 0;
};

// Published immutable input/evidence for a subsequent explicit conservative
// request. No logical or physical commands are inserted here. Payload contracts,
// original cuts, and ready scalar expressions are qualified independently of
// numerical counts, exact geometry, storage selectors, and resource assignment.
// The existing execution preconditions for dynamic extents/offsets still apply.
class ConservativeSyncAnalysis {
 public:
  const FunctionPtr& Function() const { return function_; }
  const std::vector<ConservativeSyncSite>& Sites() const { return sites_; }
  const std::vector<ConservativeSyncChildEntry>& ChildEntries() const { return children_; }
  const AutomaticSyncChildren& ChildSelection() const { return selection_; }
  const ConservativeAccessSet& ExitEffects() const { return exit_; }
  const ConservativeSyncCost& Cost() const { return cost_; }
  const std::vector<ConservativeStorageDomain>& StorageDomains() const { return domains_; }
  bool HasIncomingDiagnostics() const { return incoming_diagnostics_; }
  bool SuppliedDisjointGM() const { return disjoint_gm_; }
  bool HasRefreshKill() const { return refresh_kill_; }

 private:
  friend ConservativeSyncAnalysis AnalyzeConservativeSync(
      const FunctionPtr&, const std::string&, const std::vector<std::shared_ptr<const NativeSyncChild>>&,
      const std::vector<std::shared_ptr<const EventfulSyncChild>>&, bool, bool, bool,
      const SyncParameterBounds&, const std::optional<std::pair<int64_t, int64_t>>&, bool,
      const ImportedLocalSync*, bool);
  ConservativeSyncAnalysis(FunctionPtr function, std::vector<ConservativeSyncSite> sites,
                           ConservativeAccessSet exit, ConservativeSyncCost cost,
                           std::vector<ConservativeSyncChildEntry> children,
                           std::vector<ConservativeStorageDomain> domains, bool incoming_diagnostics,
                           bool disjoint_gm, bool refresh_kill, AutomaticSyncChildren selection)
      : function_(std::move(function)),
        sites_(std::move(sites)),
        exit_(std::move(exit)),
        cost_(cost),
        children_(std::move(children)),
        domains_(std::move(domains)),
        incoming_diagnostics_(incoming_diagnostics),
        disjoint_gm_(disjoint_gm),
        refresh_kill_(refresh_kill),
        selection_(std::move(selection)) {}
  FunctionPtr function_;
  std::vector<ConservativeSyncSite> sites_;
  ConservativeAccessSet exit_;
  ConservativeSyncCost cost_;
  std::vector<ConservativeSyncChildEntry> children_;
  std::vector<ConservativeStorageDomain> domains_;
  bool incoming_diagnostics_;
  bool disjoint_gm_;
  bool refresh_kill_;
  AutomaticSyncChildren selection_;
};

// May-dataflow over the original structured IR with certified full-overwrite KILL
// (when enabled) and joint debt/relay certification. Loops are solved via monotone
// structured CFG fixed-point without unfolding trip counts.
ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target = "a2a3",
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children = {},
    const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful_children = {},
    bool retain_incoming_diagnostics = false, bool automatic_children = false,
    bool enable_refresh_kill = true, const SyncParameterBounds& parameters = {},
    const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt, bool enable_loop_prefixes = true,
    const ImportedLocalSync* imported = nullptr, bool enable_precise_captures = false);
// Explicit policy construction. Acknowledgment-induced order is represented
// here in logical tokens before LowerSync; no physical resource is assigned.
FunctionPtr InsertConservativePrefixes(const ConservativeSyncAnalysis& analysis,
                                       const std::vector<EarlierReaderRelease>& releases = {});
bool CertifyEarlierPeerPrefixes(const ConservativeSyncAnalysis& baseline, const ImportedLocalSync& imported,
                                const std::string& target, const SyncParameterBounds& parameters,
                                const std::vector<EarlierPeerPrefix>& moves);
std::optional<PhysicalReaderReleasePlan> TryEarlierReaderReleases(const PhysicalSyncRealization& baseline,
                                                                  const ConservativeSyncAnalysis& analysis,
                                                                  const ImportedLocalSync& imported,
                                                                  const std::string& target,
                                                                  const SyncNativeReservations& reserved,
                                                                  const SyncParameterBounds& parameters);
bool OnlyAdvancesReaderRelease(const FunctionPtr& baseline, const FunctionPtr& candidate);
FunctionPtr MoveEarlierPeerPrefixes(const FunctionPtr& physical, const ConservativeSyncAnalysis& baseline,
                                    const ImportedLocalSync& imported, const std::string& target,
                                    const SyncParameterBounds& parameters,
                                    const std::vector<EarlierPeerPrefix>& prior_moves = {});
CompletionSummary SummarizeStraightLineLogicalCompletion(const StmtPtr& body, FunctionType func_type,
                                                         const std::string& target);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_DATAFLOW_H_
