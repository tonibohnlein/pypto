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

#include <map>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
struct ScopedTokenWork {
  // Read() visits include enclosing nodes; nested wait-only traversal is
  // counted separately at its wait leaves, not as additional Read() visits.
  size_t statements = 0, waits_inspected = 0, inactive_waits = 0, families = 0;
};
// Interpret actual SSA token uses in original constant frames. Returned IDs are
// empty: these are handoff identities/domains, not a physical certificate.
// Precondition: logical SSA/schema is verified; input is imported from the
// owner returned by StripStructuredLogicalSync and stripped_loops is its map.
// Families may have clipped domains and nonadjacent histories. Their existence
// establishes neither translation-invariant order nor legal physical reuse.
std::vector<PhysicalSyncFamily> ParseScopedTokenFamilies(
    const FunctionPtr& function, const CompactSyncInput& input,
    const std::map<const ForStmt*, ForStmtPtr>& stripped_loops, ScopedTokenWork* work = nullptr);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_TOKEN_FAMILIES_H_
