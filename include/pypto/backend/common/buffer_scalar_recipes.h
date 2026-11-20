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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_SCALAR_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_SCALAR_RECIPES_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "pypto/ir/expr.h"
#include "pypto/ir/type.h"

namespace pypto::backend {
// Geometry for the selected native GetValue/SetValue family. Missing means
// unsupported/unproved, never an empty memory effect. No native target, scalar
// readiness or allocation ownership is inferred from this descriptor proof.
std::optional<uint32_t> BufferScalarPointOffset(const ir::BufferTypePtr& buffer, const ir::ExprPtr& indices);
bool SupportsLogicalBufferScalar(const ir::CallPtr& call);
// Packed rank-one or rank-two ABI for point staging. This proves geometry, not input
// address validity, external publication, alias disjointness or scalar readiness.
bool IsBufferScalarGMSource(const ir::TensorTypePtr& tensor);
// Uncached behavior is not assumed. The native AIC pointer read retains the
// original caller's validity precondition and needs read-only alias evidence
// over the invocation before PyPTO may own local synchronization.
bool SupportsBufferGMRead(const std::vector<ir::ExprPtr>& args);
ir::MakeTuplePtr MakeBufferScalarGMExtent(const ir::TensorType& tensor, const ir::Span& span);
// Direction is part of the contract: load reads four GM bytes; store writes
// exactly four GM bytes. Both retain a complete aligned 32-byte local carrier.
bool SupportsBufferScalarTransfer(const std::vector<ir::ExprPtr>& args, bool load);
bool SupportsLogicalBufferScalarTransfer(const ir::CallPtr& call);
bool SupportsBufferVector1DTransfer(const std::vector<ir::ExprPtr>& args, bool load);
bool SupportsLogicalVector1DBufferTransfer(const ir::CallPtr& call);
// Native aligned DMA may write a dummy tail inside the complete private
// carrier. Its sound write enclosure is 32 bytes; only one GM element is read.
constexpr uint64_t kBufferScalarCarrierBytes = 32;
// tile.write mutates argument zero and returns that same storage. Final
// placement must preserve the complete window, including rotating-slot state.
bool LogicalScalarWriteSharesStorage(const ir::CallPtr& call, const ir::TypePtr& result_type);
}  // namespace pypto::backend
#endif  // PYPTO_BACKEND_COMMON_BUFFER_SCALAR_RECIPES_H_
