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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_ROTATING_SYNC_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_ROTATING_SYNC_ANALYSIS_H_
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
CompactSyncResult AnalyzeRotatingSync(CompactSyncInput input);
CompactSegmentResult AnalyzeRotatingSegment(const CompactSyncInput& input, size_t region, int64_t count,
                                            CompactSyncResult& work);
PointBoundary RotatingBoundary(const CompactSyncInput& input, const CompactSegmentResult& segment,
                               size_t storage, int64_t slot, int64_t row, int64_t column, size_t region = 0);
}  // namespace pypto::ir::sync
#endif  // PYPTO_IR_TRANSFORMS_UTILS_ROTATING_SYNC_ANALYSIS_H_
