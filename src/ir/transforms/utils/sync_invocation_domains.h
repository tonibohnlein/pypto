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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_INVOCATION_DOMAINS_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_INVOCATION_DOMAINS_H_

#include <utility>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Analytic exhaustive activation intervals. This service constructs no demands
// or executable protocol; callers still qualify each entire interval.
struct SyncInvocationDomains {
  std::vector<std::pair<int64_t, int64_t>> intervals;
  size_t coordinate_nodes = 0, coordinate_pieces = 0;
};
SyncInvocationDomains PartitionSyncInvocation(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                              std::pair<int64_t, int64_t> blocks);
ExprPtr DominatingSyncBlock(const FunctionPtr& owner);
ExprPtr SyncInvocationGuard(const ExprPtr& block, std::pair<int64_t, int64_t> all,
                            std::pair<int64_t, int64_t> selected, const Span& span);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_INVOCATION_DOMAINS_H_
