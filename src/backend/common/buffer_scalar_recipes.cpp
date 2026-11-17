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

#include "pypto/backend/common/buffer_scalar_recipes.h"

#include <cstddef>
#include <limits>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/structural_comparison.h"

namespace pypto::backend {
std::optional<uint32_t> BufferScalarPointOffset(const ir::BufferTypePtr& buffer, const ir::ExprPtr& indices) {
  if (!IsAlignedDenseVecBufferGeometry(buffer) ||
      (buffer->dtype_ != DataType::FP32 && buffer->dtype_ != DataType::INT32))
    return std::nullopt;
  const auto tuple = ir::As<ir::MakeTuple>(indices);
  if (!tuple || tuple->elements_.size() != 2) return std::nullopt;
  int64_t point[2];
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto index = ir::As<ir::ConstInt>(tuple->elements_[axis]);
    if (!index || index->value_ < 0 || buffer->valid_shape_[axis] < 0 ||
        index->value_ >= buffer->valid_shape_[axis])
      return std::nullopt;
    point[axis] = index->value_;
  }
  const __int128 flat = __int128(point[0]) * buffer->shape_[1] + point[1];
  if (flat > std::numeric_limits<uint32_t>::max()) return std::nullopt;
  return static_cast<uint32_t>(flat);
}

bool SupportsLogicalBufferScalar(const ir::CallPtr& call) {
  const bool read = ir::IsOp(call, "tile.read");
  if ((!read && !ir::IsOp(call, "tile.write")) || call->args_.size() != (read ? 2 : 3) ||
      !call->kwargs_.empty())
    return false;
  for (const auto& arg : call->args_)
    if (!arg) return false;
  const auto descriptor = StaticLogicalBufferDescriptor(call->args_[0]->GetType());
  if (!BufferScalarPointOffset(descriptor, call->args_[1])) return false;
  if (read) {
    const auto scalar = ir::As<ir::ScalarType>(call->GetType());
    return scalar && scalar->dtype_ == descriptor->dtype_;
  }
  const auto scalar = ir::As<ir::ScalarType>(call->args_[2]->GetType());
  const auto result = StaticLogicalBufferDescriptor(call->GetType());
  return scalar && scalar->dtype_ == descriptor->dtype_ && result && ir::structural_equal(descriptor, result);
}

bool LogicalScalarWriteSharesStorage(const ir::CallPtr& call, const ir::TypePtr& result_type) {
  if (!SupportsLogicalBufferScalar(call) || !ir::IsOp(call, "tile.write")) return false;
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto result = ir::As<ir::TileType>(result_type);
  const auto source_descriptor = StaticLogicalBufferDescriptor(source);
  const auto result_descriptor = StaticLogicalBufferDescriptor(result);
  if (!source_descriptor || !result_descriptor || !ir::structural_equal(source_descriptor, result_descriptor))
    return false;
  const auto a = source ? source->memref_.value_or(nullptr) : nullptr;
  const auto b = result ? result->memref_.value_or(nullptr) : nullptr;
  // Full MemRef equality retains base identity, byte range, slots and their
  // original selection coordinates. Base equality alone is insufficient.
  return a && b && (a == b || ir::structural_equal(a, b));
}
}  // namespace pypto::backend
