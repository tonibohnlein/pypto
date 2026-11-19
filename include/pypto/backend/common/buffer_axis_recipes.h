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

#ifndef PYPTO_BACKEND_COMMON_BUFFER_AXIS_RECIPES_H_
#define PYPTO_BACKEND_COMMON_BUFFER_AXIS_RECIPES_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/ir/expr.h"

namespace pypto::backend {

/// Axis reductions and broadcasts share explicit destination handling. Row
/// reductions additionally write scratch; column reductions here are sequential
/// and take no scratch. The axis is the reduced/broadcast matrix dimension.
struct BufferAxisRecipe {
  const char* logical_op;
  const char* buffer_op;
  const char* native_op;
  bool reduction;
  int axis = 1;

  [[nodiscard]] bool HasScratch() const { return reduction && axis == 1; }
  [[nodiscard]] size_t InputCount() const { return reduction && axis == 0 ? 1 : 2; }
};

const std::vector<BufferAxisRecipe>& GetBufferAxisRecipes();
const BufferAxisRecipe* FindLogicalBufferAxisRecipe(const std::string& name);
const BufferAxisRecipe* FindBufferAxisRecipe(const std::string& name);
bool SupportsLogicalBufferAxis(const BufferAxisRecipe& recipe, const ir::CallPtr& call);
void ValidateBufferAxisOperands(const BufferAxisRecipe& recipe, const std::vector<ir::ExprPtr>& args);

/// A2/A3 binary column broadcasts share a native implementation. A negative
/// native_valid_cols denotes the mutable valid-column template parameter.
const char* A2A3ColumnBroadcastEncodingError(DataType dtype, int64_t rows, int64_t cols,
                                             int64_t valid_row_bound, int64_t native_valid_cols,
                                             int64_t max_row_stride);
void ValidateA2A3BufferAxisEncoding(const BufferAxisRecipe& recipe, const ir::CallPtr& call);

}  // namespace pypto::backend

#endif  // PYPTO_BACKEND_COMMON_BUFFER_AXIS_RECIPES_H_
