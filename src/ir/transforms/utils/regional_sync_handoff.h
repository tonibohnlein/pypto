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

#ifndef SRC_IR_TRANSFORMS_UTILS_REGIONAL_SYNC_HANDOFF_H_
#define SRC_IR_TRANSFORMS_UTILS_REGIONAL_SYNC_HANDOFF_H_

#include <map>
#include <string>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"

namespace pypto::ir::sync {

struct PhysicalSyncRealization;

enum class HandoffReplayTrip { ZeroOrOne, ExactlyOne };

// Only generated parent commands at contiguous entry cuts between retained
// children are candidates. The caller must own terminal invocation completion.
// Trip facts are keyed by the emitted loops whose original bounds were proved.
void ReduceRegionalSyncHandoffs(PhysicalSyncRealization& result, const std::string& target,
                                const std::vector<std::vector<StmtPtr>>& handoffs,
                                const std::map<const ForStmt*, HandoffReplayTrip>& single_visit_loops);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_REGIONAL_SYNC_HANDOFF_H_
