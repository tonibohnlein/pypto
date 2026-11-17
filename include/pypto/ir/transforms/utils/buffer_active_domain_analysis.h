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

#ifndef PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACTIVE_DOMAIN_ANALYSIS_H_
#define PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACTIVE_DOMAIN_ANALYSIS_H_

#include <cstddef>
#include <utility>

#include "pypto/ir/function.h"

namespace pypto::ir {

// Optional per-request diagnostic sink. These counters are not proof state,
// do not alter admission, and count source occurrences/DAG identities rather
// than dynamic visits. They are reset on every normalization request.
struct BufferActiveDomainWork {
  size_t definition_statements = 0;
  size_t domain_statements = 0;
  size_t metadata_statements = 0;
  size_t metadata_expression_states = 0;
  size_t metadata_types = 0;
  size_t scalar_stability_nodes = 0;
  size_t rewritten_statements = 0;
  size_t rewrite_expression_dispatches = 0;
};

// A qualified candidate, not permission to publish a partially converted kernel.
// Selection must run every ordinary Buffer admission check on Candidate() and
// retain Original() on refusal. The proof is owned by these immutable functions;
// lowering consumes explicit aliases/types, never a cached permissive type view.
class BufferActiveDomainCandidate {
 public:
  const FunctionPtr& Original() const { return original_; }
  const FunctionPtr& Candidate() const { return candidate_; }

 private:
  friend BufferActiveDomainCandidate NormalizeBufferActiveDomains(const FunctionPtr&,
                                                                  BufferActiveDomainWork*);
  BufferActiveDomainCandidate(FunctionPtr original, FunctionPtr candidate)
      : original_(std::move(original)), candidate_(std::move(candidate)) {}
  FunctionPtr original_, candidate_;
};

// Qualify one common active prefix through FP32 lane-wise operations, row
// reductions, contiguous vector axis maps and counted-loop accumulation. An
// observable full result, unknown metadata observer or noninvariant carry bound
// retains the original function. No payload arithmetic, stores, physical shape
// or loop counts are narrowed. Whole-function Buffer admission is still needed.
BufferActiveDomainCandidate NormalizeBufferActiveDomains(const FunctionPtr& function,
                                                         BufferActiveDomainWork* work = nullptr);

}  // namespace pypto::ir
#endif  // PYPTO_IR_TRANSFORMS_UTILS_BUFFER_ACTIVE_DOMAIN_ANALYSIS_H_
