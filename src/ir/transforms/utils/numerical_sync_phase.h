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

#ifndef SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_PHASE_H_
#define SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_PHASE_H_

#include <map>
#include <memory>

#include "src/ir/transforms/utils/numerical_sync_merge.h"

namespace pypto::ir::sync {
// Qualified finite phase specialization inside the numerical regional service.
// Owners retain original source identities; only a finite microbody is sampled
// under proved modular phases. No repeated invocation domain is unfolded.
void PlanNumericalPhases(const SyncRegion& source, CompactSyncResult& result,
                         NumericalSyncHierarchy& hierarchy);
void BuildNumericalPhaseCells(CompactSyncResult& result, NumericalSyncHierarchy& hierarchy);
std::map<size_t, std::shared_ptr<const NumericalNode>> BuildNumericalPhaseProviders(
    CompactSyncResult& result, NumericalSyncHierarchy& hierarchy);
}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_NUMERICAL_SYNC_PHASE_H_
