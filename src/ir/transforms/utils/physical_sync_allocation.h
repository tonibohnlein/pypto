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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_

#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
using SyncDirection = std::pair<PipeType, PipeType>;
using SyncNativeReservations = std::map<SyncDirection, std::set<int>>;

// Qualified direct-cut order provider. The command inventory is independent
// of minimum-demand analysis: token clients retain every actual use. Callbacks
// are consumed synchronously while their immutable owner is alive.
struct DirectSyncOrder {
  // Sufficient C(source) -> I(target) proof, not generic event precedence.
  std::function<bool(const Occurrence&, const Occurrence&)> precedes;
  std::function<bool(const Occurrence&, const Occurrence&)> reference_before;
  // Minimum uniform C(source)->I(target) displacement within one region.
  std::function<std::optional<__int128>(size_t, size_t, size_t)> threshold;
  std::function<bool(size_t)> scoped_reuse, scoped_barrier;
  bool finite_minimum = false;
};
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input,
                                          std::vector<PhysicalSyncFamily> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted);

// Separate allocation from shared cut application for mutually exclusive cases.
PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input,
                                           std::vector<PhysicalSyncFamily> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted);
struct AssignedSyncCase {
  const CompactSyncInput* input;
  PhysicalSyncRealization* plan;
};
// Borrowed owners live through this synchronous call. All inputs must refer to
// the same original owner and exhaust its supplied invocation domain.
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases);

// Internal emitter utility, not a general IR canonicalizer. Inputs contain
// only physical command leaves, sequences and guards over ready immutable
// original coordinates; no definitions, payloads or effectful conditions.
// Counters must already include the input guards. Else-bearing trees are opaque.
std::vector<StmtPtr> ShareSyncCommandGuards(const std::vector<StmtPtr>& commands,
                                            PhysicalSyncRealization& counters);
struct SyncCommandCase {
  std::pair<int64_t, int64_t> blocks;
  StmtPtr body;
};
// possible is a normalized conservative upper domain of the original cut.
// Use the entire launch domain when evidence is unavailable, never an empty list.
// Like ShareSyncCommandGuards, bodies contain only ready, immutable-coordinate
// guards and physical commands. Factoring preserves each case's ordered trace;
// no definition/payload may move and assignments are never recolored.
// uniform_variables is optional original-cut evidence: only literal-bound,
// positive-step IVs with a launch-independent domain at this cut may be named.
// Omit it when original control or nested bounds do not establish that fact.
std::vector<StmtPtr> EmitSyncCommandCases(const ExprPtr& block,
                                          const std::vector<std::pair<int64_t, int64_t>>& possible,
                                          const std::vector<SyncCommandCase>& cases,
                                          PhysicalSyncRealization& work, const Span& span,
                                          const std::set<const Var*>& uniform_variables = {});
// Same emitter-only precondition as ShareSyncCommandGuards. The owner supplies
// name reservations; definitions stay inside the original command scope.
StmtPtr MaterializeSyncCommandValues(const FunctionPtr& owner, const StmtPtr& commands);

// Publish only an established complete physical owner (literal SET and WAIT),
// or an explicit delegation outcome. This does not qualify or allocate a plan.
FunctionPtr SyncOutcome(const FunctionPtr& function, bool pypto, const std::string& reason);

// Shared native-owner and target eligibility policy. Neither helper computes
// demands or assumes that a logical token owns a consuming hardware event.
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target);
std::vector<int> SyncEligibleIds(const CompactSyncInput& input, const std::string& target,
                                 SyncDirection direction, const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed);

// Existing monotone lifetime-chain sweep. Lanes must already be sorted by
// publication. For one direction, consumptions form a chain and each legal
// reuse set is a publication suffix. The callbacks retain their proof owner.
// O(L log L) ordering plus O(L) reuse queries; query cost is charged separately.
std::vector<int> ColorOrderedSyncLanes(size_t count, SyncDirection direction,
                                       const std::function<bool(size_t, size_t)>& wait_before,
                                       const std::function<bool(size_t, size_t)>& reusable,
                                       const std::vector<int>& eligible, bool finite_minimum);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_ALLOCATION_H_
