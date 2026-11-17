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
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
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
  ExprPtr original_root;  // GM provenance qualified by the supplied disjoint contract.
};
using ConservativeAccessSet = std::set<ConservativeAccessFact>;

struct ConservativeSyncSite {
  StmtPtr cut;  // Original statement, immediately before its payload call.
  backend::BufferSyncOperation operation;
  ConservativeAccessSet effects;
  ConservativeAccessSet incoming;  // Unavailable unless HasIncomingDiagnostics() is true.
  // Full-overwrite (space, domain) pairs proven to overwrite the entire domain cleanly.
  std::vector<std::pair<MemorySpace, size_t>> refreshed_domains;
  // Baseline storage obligations and the ordered selected subset. Non-storage
  // child/rearming services are protected separately and never filtered here.
  std::vector<PipeType> source_pipes;
  std::vector<PipeType> required_source_pipes;
};

struct ConservativeSyncChildEntry {
  ForStmtPtr scope;
  std::shared_ptr<const NativeSyncChild> child;
  ConservativeAccessSet incoming, effects;  // incoming requires HasIncomingDiagnostics().
  // A child has several participating pipes, never one fabricated payload pipe.
  std::vector<std::pair<PipeType, PipeType>> services;
  std::shared_ptr<const EventfulSyncChild> eventful;
  std::vector<std::pair<PipeType, PipeType>> required_services;
  CompletionSummary completion = CompletionSummary::Identity();
  bool first_visit = false;
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
  // Diagnostic, output-sensitive O(depth) walk of the shared original-parent
  // index. Production sites retain only cuts, never one copied path per site.
  std::vector<StmtPtr> ControlPath(size_t site) const;

 private:
  friend ConservativeSyncAnalysis AnalyzeConservativeSync(
      const FunctionPtr&, const std::string&, const std::vector<std::shared_ptr<const NativeSyncChild>>&,
      const std::vector<std::shared_ptr<const EventfulSyncChild>>&, bool, bool, bool,
      const SyncParameterBounds&, const std::optional<std::pair<int64_t, int64_t>>&);
  ConservativeSyncAnalysis(FunctionPtr function, std::vector<ConservativeSyncSite> sites,
                           ConservativeAccessSet exit, ConservativeSyncCost cost,
                           std::shared_ptr<const ParentStmtAnalysis> parents,
                           std::vector<ConservativeSyncChildEntry> children,
                           std::vector<ConservativeStorageDomain> domains, bool incoming_diagnostics,
                           bool disjoint_gm, bool refresh_kill, AutomaticSyncChildren selection)
      : function_(std::move(function)),
        sites_(std::move(sites)),
        exit_(std::move(exit)),
        cost_(cost),
        parents_(std::move(parents)),
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
  std::shared_ptr<const ParentStmtAnalysis> parents_;
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
    const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt);
// Explicit policy construction. Acknowledgment-induced order is represented
// here in logical tokens before LowerSync; no physical resource is assigned.
FunctionPtr InsertConservativePrefixes(const ConservativeSyncAnalysis& analysis);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_DATAFLOW_H_
