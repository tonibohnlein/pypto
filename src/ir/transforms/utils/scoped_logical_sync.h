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

#ifndef SRC_IR_TRANSFORMS_UTILS_SCOPED_LOGICAL_SYNC_H_
#define SRC_IR_TRANSFORMS_UTILS_SCOPED_LOGICAL_SYNC_H_

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {
// Existing scoped demand export -> original-frame RFC carries/captures. This
// constructs logical facts only; it does not certify physical flag reuse.
LogicalSyncRealization RealizeScopedLogicalSync(const CompactSyncResult& analysis);
LogicalSyncRealization RealizeLogicalInvocation(const SyncInvocationAnalysis& analysis);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_LOGICAL_SYNC_H_
