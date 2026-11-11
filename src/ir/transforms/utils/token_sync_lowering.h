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

#include <optional>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
bool HasLogicalSync(const FunctionPtr& function);
// Interpret actual logical token uses. This finite adapter does not compute or
// substitute minimum demands. families is empty: token cuts are not original
// payload occurrence pairs; function and command/resource counters are valid.
PhysicalSyncRealization LowerLogicalSync(const FunctionPtr& function, const std::string& target,
                                         const std::optional<std::vector<int>>& eligible_ids);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_TOKEN_SYNC_LOWERING_H_
