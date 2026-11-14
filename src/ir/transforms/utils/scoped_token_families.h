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

#ifndef SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_FAMILIES_H_
#define SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_FAMILIES_H_

#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
struct ScopedTokenWork {
  // Read() visits include enclosing nodes; nested wait-only traversal is
  // counted separately at its wait leaves, not as additional Read() visits.
  size_t statements = 0, waits_inspected = 0, inactive_waits = 0, families = 0;
  size_t equation_visits = 0, capture_specializations = 0;
  size_t union_candidates = 0, union_merges = 0, union_passes = 0;
  size_t union_reuse_checks = 0, union_reuse_refusals = 0;
};
// One immutable selected direct protocol. Raw use IDs index this inventory;
// physical preparation may group uses but cannot reorder or erase them.
struct ScopedProtocolCut {
  size_t region, site;
  std::vector<size_t> before, after;
};
class ScopedSyncProtocol {
 public:
  const FunctionPtr& TokenOwner() const { return tokens_; }
  const CompactSyncInput& Input() const { return input_; }
  const std::vector<PhysicalSyncFamily>& Families() const { return families_; }
  const std::vector<ScopedProtocolCut>& Cuts() const { return cuts_; }
  void ValidateFor(const CompactSyncInput& input) const;
  void ValidateTokens(const FunctionPtr& tokens) const;

 private:
  FunctionPtr tokens_;
  CompactSyncInput input_;
  std::vector<PhysicalSyncFamily> families_;
  std::vector<ScopedProtocolCut> cuts_;
  ScopedSyncProtocol(FunctionPtr tokens, CompactSyncInput input, std::vector<PhysicalSyncFamily> families);
  friend std::shared_ptr<const ScopedSyncProtocol> ParseScopedTokenProtocol(
      const FunctionPtr&, CompactSyncInput, const std::map<const ForStmt*, ForStmtPtr>&, ScopedTokenWork*);
};
// Interpret actual SSA token uses in original constant frames. Returned IDs are
// empty: these are handoff identities/domains, not a physical certificate.
// Precondition: logical SSA/schema is verified; input is imported from the
// owner returned by StripStructuredLogicalSync and stripped_loops is its map.
// Families may have clipped domains and nonadjacent histories. Their existence
// establishes neither translation-invariant order nor legal physical reuse.
std::shared_ptr<const ScopedSyncProtocol> ParseScopedTokenProtocol(
    const FunctionPtr& function, CompactSyncInput input,
    const std::map<const ForStmt*, ForStmtPtr>& stripped_loops, ScopedTokenWork* work = nullptr);
// Precondition: families are the exact unassigned output of the parser above
// for this input. This is not admission for arbitrary user-supplied domains.
// Exact one-axis unions of disjoint actual-token records, admitted only when
// the caller proves reuse for the candidate using its existing order service.
// Failed proofs retain both original families; overlapping records are never
// deduplicated. This does not change parsing or select a different protocol.
std::vector<PhysicalSyncFamily> NormalizeScopedTokenFamilies(
    const CompactSyncInput& input, std::vector<PhysicalSyncFamily> families,
    const std::function<bool(const PhysicalSyncFamily&)>& can_reuse, ScopedTokenWork* work = nullptr);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_FAMILIES_H_
