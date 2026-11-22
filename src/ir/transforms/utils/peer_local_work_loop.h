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
#include <cstdint>
#include <map>
#include <set>
#include <utility>
#include <vector>

#include "src/ir/transforms/utils/peer_local_effects.h"

namespace pypto::ir::sync {

// Conservative conflicts are unordered; realized boundary commands are directed
// and kept in execution order. A self direction denotes a local barrier.
using PeerWorkPairs = std::set<SyncDirection>;
using PeerWorkRange = std::pair<size_t, size_t>;
using PeerWorkSeamIds = std::map<SyncDirection, int>;

struct PeerWorkAccess {
  int64_t modulus, begin, end;
};

// Work-loop effects are byte intervals within a storage slot. Comparing
// residues modulo this orbit also covers earlier items more than one trip away.
PeerWorkAccess ProjectPeerWorkAccess(const CompactSyncInput& input, const PeerLocalWorkLoop& loop,
                                     const SyncAccess& access);

struct PeerWorkPlacement {
  const PeerLocalWorkLoop* loop;
  std::vector<SyncDirection> entry, exit;
};

void AddPeerWorkConflicts(const CompactSyncInput& input, const PeerLocalWorkLoop& loop, PeerWorkRange earlier,
                          PeerWorkRange later, PeerWorkPairs& pairs);
std::vector<SyncDirection> ClosePeerWorkPairs(const PeerWorkPairs& pairs);
void SelectPeerWorkSeams(const CompactSyncInput& input, const PeerLocalRegion& region,
                         const std::vector<const PhysicalSyncFamily*>& families, const PeerWorkSeamIds& ids,
                         bool scalar_serial, PeerWorkPlacement& placement);
// Copy only literal child visits inside the verified work item. The selected
// local sync guards are constant at every original coordinate for each copy.
FunctionPtr SpecializePeerWorkGuards(const FunctionPtr& owner, const std::set<const Var*>& work_ivs);
FunctionPtr AddPeerWorkSeams(const FunctionPtr& function, const std::vector<PeerWorkPlacement>& placements,
                             const PeerWorkSeamIds& ids);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_WORK_LOOP_H_
