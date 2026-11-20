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

#ifndef SRC_IR_TRANSFORMS_UTILS_PEER_FIFO_PAIR_H_
#define SRC_IR_TRANSFORMS_UTILS_PEER_FIFO_PAIR_H_

#include <map>
#include <string>
#include <vector>

#include "pypto/ir/program.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {

// The backend has one switch for local insertion per emitted translation unit.
struct PeerLocalDecision {
  FunctionPtr function;  // Original on delegation, physically synchronized on publication.
  std::string fallback_reason;
};

std::vector<PeerLocalDecision> PlanPeerLocalUnits(const ProgramPtr& program,
                                                  const SyncInvocationEvidence& invocation);

// Explicit diagnostic for the authored protocol. RealizeSync does not call it.
std::map<std::string, std::string> InspectPeerFifoProtocols(const ProgramPtr& program);

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_PEER_FIFO_PAIR_H_
