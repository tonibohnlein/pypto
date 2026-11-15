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
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/parent_stmt_analysis.h"
#include "src/ir/transforms/utils/retained_sync_child.h"

namespace pypto::ir::sync {

// A whole-memory-space enclosure, not an exact footprint or selector. Different
// roots/views/slots in one space intentionally share a class. No disjoint-GM
// premise is used. write=false denotes a possible read, write=true a write.
using ConservativeAccessFact = std::tuple<MemorySpace, PipeType, bool>;
using ConservativeAccessSet = std::set<ConservativeAccessFact>;

struct ConservativeSyncSite {
  StmtPtr cut;  // Original statement, immediately before its payload call.
  backend::BufferSyncOperation operation;
  ConservativeAccessSet effects;
  ConservativeAccessSet incoming;
  std::vector<PipeType> source_pipes;
};

struct ConservativeSyncChildEntry {
  ForStmtPtr scope;
  std::shared_ptr<const NativeSyncChild> child;
  ConservativeAccessSet incoming, effects;
  // A child has several participating pipes, never one fabricated payload pipe.
  std::vector<std::pair<PipeType, PipeType>> services;
};

struct ConservativeSyncCost {
  size_t statements = 0;
  size_t expression_visits = 0;
  size_t dataflow_visits = 0;
  size_t retained_facts = 0;
  size_t services = 0;
  size_t parent_links = 0;
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
  const ConservativeAccessSet& ExitEffects() const { return exit_; }
  const ConservativeSyncCost& Cost() const { return cost_; }
  // Diagnostic, output-sensitive O(depth) walk of the shared original-parent
  // index. Production sites retain only cuts, never one copied path per site.
  std::vector<StmtPtr> ControlPath(size_t site) const;

 private:
  friend ConservativeSyncAnalysis AnalyzeConservativeSync(
      const FunctionPtr&, const std::string&, const std::vector<std::shared_ptr<const NativeSyncChild>>&);
  ConservativeSyncAnalysis(FunctionPtr function, std::vector<ConservativeSyncSite> sites,
                           ConservativeAccessSet exit, ConservativeSyncCost cost,
                           std::shared_ptr<const ParentStmtAnalysis> parents,
                           std::vector<ConservativeSyncChildEntry> children)
      : function_(std::move(function)),
        sites_(std::move(sites)),
        exit_(std::move(exit)),
        cost_(cost),
        parents_(std::move(parents)),
        children_(std::move(children)) {}
  FunctionPtr function_;
  std::vector<ConservativeSyncSite> sites_;
  ConservativeAccessSet exit_;
  ConservativeSyncCost cost_;
  std::shared_ptr<const ParentStmtAnalysis> parents_;
  std::vector<ConservativeSyncChildEntry> children_;
};

// No-KILL may-dataflow over the original structured IR. GEN is computed bottom
// up. A loop body receives IN union GEN(body), and exit preserves the zero-trip
// IN. Branches join by union. This deliberately includes structurally possible
// but infeasible histories; it is not exact concrete reachability. Skipped or
// partial writes never erase obligations. Pure while conditions are checked on
// entry and every visit, including the final false test; effectful conditions
// require a separate readiness/visibility contract and are refused.
//
// Fixed spaces/pipes bound the fact universe C. State work/storage is O(N*C+E);
// conflict selection additionally scans O(N*C*C) pairs (C is target-fixed).
// Original control uses one O(N) parent index; requested paths cost their output
// depth only. E counts actual scalar DAG visits (deduplicated per readiness
// query). No trip count is unfolded. Runtime
// realization, invocation completion, and resource reuse remain separate gates.
ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target = "a2a3",
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children = {});
// Explicit policy construction. Acknowledgment-induced order is represented
// here in logical tokens before LowerSync; no physical resource is assigned.
FunctionPtr InsertConservativePrefixes(const ConservativeSyncAnalysis& analysis);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_CONSERVATIVE_SYNC_DATAFLOW_H_
