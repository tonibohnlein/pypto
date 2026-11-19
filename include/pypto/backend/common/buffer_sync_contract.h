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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_
#define PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_

#include <string>
#include <utility>
#include <vector>

#include "pypto/ir/function.h"
#include "pypto/ir/pipe.h"
#include "pypto/sync/isa_macro_resources.h"

namespace pypto::backend {
// Operation identity shared with Buffer emission. This does not qualify effects,
// target execution, aliases, readiness or synchronization ownership.
const char* BufferNativeOperation(const ir::CallPtr& call);

// Factory-owned variant evidence. Resource lookup cannot rebind a selection to
// another core/target or fabricate an audited signal-free macro.
class BufferSyncImplementation {
 public:
  ir::PipeType Pipe() const { return pipe_; }
  bool AccumulatorSizeEligible() const { return accumulator_size_eligible_; }
  std::vector<sync::IsaResourceClaim> Claims() const;

 private:
  friend BufferSyncImplementation SelectBufferSyncImplementation(const ir::CallPtr&, ir::FunctionType,
                                                                 const std::string&);
  BufferSyncImplementation(ir::PipeType pipe, std::string macro, ir::FunctionType core, std::string target)
      : pipe_(pipe), macro_(std::move(macro)), core_(core), target_(std::move(target)) {}
  ir::PipeType pipe_;
  std::string macro_;
  ir::FunctionType core_;
  std::string target_;
  bool accumulator_size_eligible_ = false;
  bool mat_to_operand_extract_ = false;
};
// Input is a registry-validated Buffer call. Select only existing single-pipe
// variants; geometry/readiness/modes remain the importer's obligation. Unknown
// targets retain conservative mathematics without accumulator protection, but
// Claims() refuses executable resource qualification for them.
BufferSyncImplementation SelectBufferSyncImplementation(const ir::CallPtr& call, ir::FunctionType core,
                                                        const std::string& target);

struct LocalSyncExecutionContract {
  bool scalar_prefix_serialized;
  const char* terminal_completion_op;
};
// A missing target contract never certifies completion or physical ownership.
const LocalSyncExecutionContract& RequireLocalSyncExecutionContract(const std::string& target);
}  // namespace pypto::backend
#endif  // PYPTO_BACKEND_COMMON_BUFFER_SYNC_CONTRACT_H_
