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

#ifndef SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_EFFECTS_H_
#define SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_EFFECTS_H_

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/cross_core_pipe.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"

namespace pypto::ir::sync {

struct PeerLocalWorkLoop {
  ForStmtPtr original;
  // Half-open segment range comprising one execution of the original body.
  size_t first, end;
  // Each receive domain advances by this many pops per work item, modulo its depth.
  std::map<size_t, int64_t> ring_advance;
};

struct PeerLocalLane {
  CompactSyncInput input;
  SyncNativeReservations reservations;
  std::optional<PeerLocalWorkLoop> work_loop;
};

struct PeerGmEffects {
  std::set<size_t> formals;
  std::set<size_t> scalar_reads;
  std::set<size_t> writes;
};

struct PeerLocalEffects {
  // One common path, or the then and else paths of the original AIV subblock branch.
  std::vector<PeerLocalLane> lanes;
  PeerGmEffects gm;
};

// Recover the native physical effects at their original statement cuts. Throws
// ValueError if any effective path, operation or placement is unqualified.
PeerLocalEffects ImportPeerLocalEffects(const FunctionPtr& function,
                                        const cross_core_pipe::PeerTransportEffects& transport,
                                        const std::string& target, const SyncParameterBounds& parameters);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_PEER_LOCAL_EFFECTS_H_
