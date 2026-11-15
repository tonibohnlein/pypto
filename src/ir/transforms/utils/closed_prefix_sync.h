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
#ifndef SRC_IR_TRANSFORMS_UTILS_CLOSED_PREFIX_SYNC_H_
#define SRC_IR_TRANSFORMS_UTILS_CLOSED_PREFIX_SYNC_H_

#include <optional>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Syntactic routing only; this does not establish matching, resource, or native
// execution qualification. It lets public lowering avoid unrelated occurrence
// bound recovery when the actual program uses only current-cut transactions.
bool HasOnlyClosedPrefixSyntax(const FunctionPtr& function);
// Establish operation/native-resource eligibility even for an empty logical
// program. This interprets no required-order edges and supplies no coverage
// claim: explicit LowerSync still consumes the caller's actual token program.
bool QualifyLocalSyncInvocation(const FunctionPtr& function, const std::string& target);
// Interpret actual four-command logical transactions at original structured
// cuts. Absence means the token language is not this class; a qualified-looking
// program with unavailable native/resource evidence refuses before any emission.
// No required-order analysis or numerical occurrence inventory is consulted.
std::optional<PhysicalSyncRealization> TryLowerClosedPrefixSync(
    const FunctionPtr& function, const std::string& target,
    const std::optional<std::vector<int>>& eligible_ids);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_CLOSED_PREFIX_SYNC_H_
