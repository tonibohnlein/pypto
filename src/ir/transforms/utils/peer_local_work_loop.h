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

#ifndef SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_WORK_LOOP_H_
#define SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_WORK_LOOP_H_

#include <cstddef>
#include <map>
#include <set>
#include <utility>

#include "src/ir/transforms/utils/peer_local_effects.h"

namespace pypto::ir::sync {

// Conflict pairs are unordered. A seam closes both directions together;
// a self pair needs current-prefix completion on that pipe only.
using PeerWorkPairs = std::set<SyncDirection>;
using PeerWorkRange = std::pair<size_t, size_t>;
using PeerWorkSeamIds = std::map<SyncDirection, int>;

void AddPeerWorkConflicts(const PeerLocalLane& lane, PeerWorkRange earlier, PeerWorkRange later,
                          PeerWorkPairs& pairs);
FunctionPtr AddPeerWorkSeams(const FunctionPtr& function, const PeerLocalWorkLoop& loop,
                             const PeerWorkPairs& entry, const PeerWorkPairs& exit,
                             const PeerWorkSeamIds& ids);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_WORK_LOOP_H_
