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

#ifndef SRC_IR_TRANSFORMS_UTILS_LOCAL_COMPLETION_RELAY_H_
#define SRC_IR_TRANSFORMS_UTILS_LOCAL_COMPLETION_RELAY_H_

#include <string>

#include "pypto/ir/function.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"

namespace pypto::ir::sync {

// First remove redundant completion services using equivalent relays, then
// separate captures that unnecessarily delay an independent consumer. Both
// steps preserve required payload order and consuming-event reuse.
FunctionPtr RefineLocalCompletionPlan(const FunctionPtr& physical, const ImportedLocalCuts& cuts,
                                      const std::string& target,
                                      const SyncNativeReservations& native_reservations);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_LOCAL_COMPLETION_RELAY_H_
