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

#ifndef SRC_IR_TRANSFORMS_UTILS_PEER_GM_INVOCATION_H_
#define SRC_IR_TRANSFORMS_UTILS_PEER_GM_INVOCATION_H_

#include <cstddef>
#include <memory>
#include <set>
#include <string>

#include "pypto/ir/program.h"

namespace pypto::ir::sync {

struct SyncInvocationEvidence;
struct PeerGmEffects;

// Checks each endpoint's separation from its compiler-allocated FIFO workspace.
// Ordinary tensor arguments may alias one another. Index caller edges once and inspect only relevant root
// bodies.
class PeerInvocationGmEvidence {
 public:
  PeerInvocationGmEvidence(const ProgramPtr& program, const SyncInvocationEvidence& invocation);
  ~PeerInvocationGmEvidence();

  const std::set<std::string>& CallersOf(const std::string& callee) const;
  void Check(const FunctionPtr& group, const std::set<size_t>& ordinary_formals, size_t workspace_formal);
  void CheckScalarReads(const FunctionPtr& group, const PeerGmEffects& effects, bool disjoint_gm) const;

 private:
  struct State;
  std::unique_ptr<State> state_;
};

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_PEER_GM_INVOCATION_H_
