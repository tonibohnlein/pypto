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

#ifndef SRC_IR_TRANSFORMS_UTILS_TOKEN_SYNC_LOWERING_H_
#define SRC_IR_TRANSFORMS_UTILS_TOKEN_SYNC_LOWERING_H_

#include <map>
#include <optional>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Current payload/control/native qualification for token lowering. This does
// not recover dependence footprints or assert a GM alias partition. Geometry
// needed to qualify a native instruction or original control remains checked.
CompactSyncInput ImportSyncPayloads(const FunctionPtr& function, const std::string& target,
                                    const SyncParameterBounds& parameters = {},
                                    const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt);

// Insertion owns its generated SSA: malformed output is a compiler error,
// never an unsupported-input reason for downstream delegation.
void CheckGeneratedLogicalSync(const FunctionPtr& function);
// Shared public/production logical stage. Rewrites belong here, never only in
// the pass wrapper. Invalid caller input is a qualification error; invalid
// output after a future rewrite is a compiler error.
FunctionPtr OptimizeLogicalSync(const FunctionPtr& function);

// Original payload/native import for token consumers. The returned loop map
// preserves correspondence to the token IR; it does not retain demand analysis.
struct StrippedLogicalSync {
  FunctionPtr function;
  std::map<const ForStmt*, ForStmtPtr> loops;
};
StrippedLogicalSync StripStructuredLogicalSync(const FunctionPtr& function, bool allow_nested);
bool HasLogicalSync(const FunctionPtr& function);
// Executable summary only. Case-specific family coordinates are never flattened
// into one misleading inventory; their immutable owners live through emission.
struct LoweredLogicalSync {
  FunctionPtr function;
  size_t set_sites = 0, wait_sites = 0;
  // Diagnostic allocation receipts stay separated by their original case.
  std::vector<PhysicalSyncRealization> physical_cases;
};
LoweredLogicalSync LowerStructuredLogicalSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids, const SyncParameterBounds& parameters = {},
    std::optional<std::pair<int64_t, int64_t>> blocks = std::nullopt, const SyncSchedulePolicy& policy = {});
// Interpret actual logical token uses without substituting minimum demands.
// The finite adapter keeps actual command cuts and has no payload families;
// canonical structured transport certifies and retains direct-cut families.
LoweredLogicalSync LowerLogicalSync(const FunctionPtr& function, const std::string& target,
                                    const std::optional<std::vector<int>>& eligible_ids,
                                    const SyncParameterBounds& parameters = {},
                                    std::optional<std::pair<int64_t, int64_t>> blocks = std::nullopt,
                                    const SyncSchedulePolicy& policy = {});
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_TOKEN_SYNC_LOWERING_H_
