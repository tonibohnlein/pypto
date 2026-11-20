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
#include <memory>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
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

bool IsBufferScalarGMSource(const ir::TensorTypePtr& tensor) {
  if (!tensor || (tensor->dtype_ != DataType::INT32 && tensor->dtype_ != DataType::FP32) ||
      !IsBufferTensorShape(tensor->shape_)) {
    return false;
  }
  const auto rank = tensor->shape_.size();
  const auto& outer_extent = tensor->shape_.front();
  if (!ir::As<ir::ConstInt>(outer_extent) && !ir::As<ir::Var>(outer_extent)) {
    return false;
  }
  const auto columns = ir::As<ir::ConstInt>(tensor->shape_.back());
  // ND TLOAD/TSTORE derive the GM burst gap from the parent row pitch,
  // even for a one-row partition. Keep that byte gap in the native field.
  constexpr int64_t kMaxNativeGap = std::numeric_limits<uint32_t>::max();
  const auto element_bytes = static_cast<int64_t>(tensor->dtype_.GetByte());
  if (rank == 2 && (!columns || columns->value_ - 1 > kMaxNativeGap / element_bytes)) {
    return false;
  }
  if (!tensor->tensor_view_) {
    return true;
  }
  const auto& view = *tensor->tensor_view_;
  if (view.layout != ir::TensorLayout::ND || view.pad != ir::PadValue::null ||
      (!view.valid_shape.empty() &&
       !ir::tile_view_semantics::ShapeExprListsEquivalent(view.valid_shape, tensor->shape_))) {
    return false;
  }
  if (view.stride.empty()) {
    return true;
  }
  if (view.stride.size() != rank) {
    return false;
  }
  const auto inner = ir::As<ir::ConstInt>(view.stride.back());
  const auto outer = ir::As<ir::ConstInt>(view.stride.front());
  return inner && inner->value_ == 1 && (rank == 1 || (outer && outer->value_ == columns->value_));
}

ir::MakeTuplePtr MakeBufferScalarGMExtent(const ir::TensorType& tensor, const ir::Span& span) {
  const auto one = std::make_shared<ir::ConstInt>(1, DataType::INDEX, span);
  return std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>(tensor.shape_.size(), one), span);
}

bool SupportsBufferGMRead(const std::vector<ir::ExprPtr>& args) {
  if (args.size() != 2 || !ir::AsVarLike(args[0])) return false;
  const auto source = ir::As<ir::TensorType>(args[0]->GetType());
  const auto indices = ir::As<ir::MakeTuple>(args[1]);
  if (!IsBufferScalarGMSource(source) || source->tensor_view_ || !indices ||
      indices->elements_.size() != source->shape_.size()) {
    return false;
  }
  const auto last = ir::As<ir::ConstInt>(indices->elements_.back());
  const auto first = ir::As<ir::ConstInt>(indices->elements_.front());
  if (first && last) {
    __int128 offset = first->value_;
    if (source->shape_.size() == 2) {
      offset = offset * ir::As<ir::ConstInt>(source->shape_[1])->value_ + last->value_;
    }
    if (offset >= std::numeric_limits<int64_t>::max() / source->dtype_.GetByte()) return false;
  }
  for (size_t axis = 0; axis < source->shape_.size(); ++axis) {
    const auto& index = indices->elements_[axis];
    const auto type = index ? ir::As<ir::ScalarType>(index->GetType()) : nullptr;
    if (!type || (type->dtype_ != DataType::INDEX && !type->dtype_.IsSignedInt())) return false;
    const auto coordinate = ir::As<ir::ConstInt>(index);
    const auto length = ir::As<ir::ConstInt>(source->shape_[axis]);
    if (coordinate && (coordinate->value_ < 0 || (length && coordinate->value_ >= length->value_)))
      return false;
  }
  return true;
}

bool SupportsBufferScalarTransfer(const std::vector<ir::ExprPtr>& args, bool load) {
  if (args.size() != 4) return false;
  for (const auto& arg : args)
    if (!arg) return false;
  const auto tensor = ir::As<ir::TensorType>(args[load ? 0 : 3]->GetType());
  const auto carrier = ir::As<ir::BufferType>(args[load ? 3 : 0]->GetType());
  const auto offsets = ir::As<ir::MakeTuple>(args[1]);
  const auto valid = ir::As<ir::MakeTuple>(args[2]);
  if (!IsBufferScalarGMSource(tensor) || !IsAlignedDenseVecBufferGeometry(carrier) ||
      carrier->dtype_ != tensor->dtype_ || carrier->shape_ != std::vector<int64_t>{1, 8} ||
      carrier->valid_shape_ != std::vector<int64_t>{1, 1} || !offsets || !valid ||
      offsets->elements_.size() != tensor->shape_.size() ||
      valid->elements_.size() != tensor->shape_.size()) {
    return false;
  }
  for (size_t axis = 0; axis < tensor->shape_.size(); ++axis) {
    const auto& offset = offsets->elements_[axis];
    const auto extent = ir::As<ir::ConstInt>(valid->elements_[axis]);
    const auto index_type = offset ? ir::As<ir::ScalarType>(offset->GetType()) : nullptr;
    if (!extent || extent->value_ != 1 || !index_type ||
        !(index_type->dtype_ == DataType::INDEX || index_type->dtype_.IsSignedInt())) {
      return false;
    }
    // Symbolic indices retain the original defined tensor.read/write precondition.
    const auto index = ir::As<ir::ConstInt>(offset);
    const auto length = ir::As<ir::ConstInt>(tensor->shape_[axis]);
    if (index && (index->value_ < 0 || (length && index->value_ >= length->value_))) {
      return false;
    }
  }
  return true;
}

bool SupportsLogicalBufferScalarTransfer(const ir::CallPtr& call) {
  if (ir::IsOp(call, "tile.store")) {
    if (call->args_.size() != 3 || !call->args_[0] || !call->args_[1] || !call->args_[2]) return false;
    const auto descriptor = StaticLogicalBufferDescriptor(call->args_[0]->GetType());
    const auto tensor = ir::As<ir::TensorType>(call->args_[2]->GetType());
    if (!descriptor || !IsBufferScalarGMSource(tensor)) {
      return false;
    }
    const auto carrier = std::make_shared<ir::Var>("scalar_carrier_geometry", descriptor, call->span_);
    // Vec-to-GM ND TSTORE has one four-byte burst for this valid unit view.
    // This is the store geometry, not authorization from the TLOAD contract.
    return SupportsBufferScalarTransfer(
        {carrier, call->args_[1], MakeBufferScalarGMExtent(*tensor, call->span_), call->args_[2]}, false);
  }
  if (!ir::IsOp(call, "tile.load") || call->args_.size() != 4) return false;
  const auto shape = ir::As<ir::MakeTuple>(call->args_[2]);
  const auto tensor = call->args_[0] ? ir::As<ir::TensorType>(call->args_[0]->GetType()) : nullptr;
  const auto descriptor = StaticLogicalBufferDescriptor(call->GetType());
  if (!IsBufferScalarGMSource(tensor) || !shape || shape->elements_.size() != tensor->shape_.size() ||
      !descriptor) {
    return false;
  }
  const auto capacity = ir::As<ir::ConstInt>(shape->elements_.back());
  const auto rows = ir::As<ir::ConstInt>(shape->elements_.front());
  if (!capacity || capacity->value_ != 8 || (tensor->shape_.size() == 2 && (!rows || rows->value_ != 1))) {
    return false;
  }
  const auto carrier = std::make_shared<ir::Var>("scalar_carrier_geometry", descriptor, call->span_);
  return SupportsBufferScalarTransfer({call->args_[0], call->args_[1], call->args_[3], carrier}, true);
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

bool SupportsBufferVector1DTransfer(const std::vector<ir::ExprPtr>& args, bool load) {
  if (args.size() != 4) return false;
  for (const auto& arg : args)
    if (!arg) return false;
  const auto tensor = ir::As<ir::TensorType>(args[load ? 0 : 3]->GetType());
  const auto buffer = ir::As<ir::BufferType>(args[load ? 3 : 0]->GetType());
  const auto offsets = ir::As<ir::MakeTuple>(args[1]);
  const auto valid = ir::As<ir::MakeTuple>(args[2]);
  if (!tensor || tensor->shape_.size() != 1 || !IsBufferTensorShape(tensor->shape_) ||
      !IsDenseBufferTransferDtype(tensor->dtype_) || !buffer || !IsAlignedDenseVecBufferGeometry(buffer) ||
      buffer->shape_.size() != 2 || buffer->shape_[0] != 1 || buffer->shape_[1] <= 1 ||
      buffer->dtype_ != tensor->dtype_ || !offsets || offsets->elements_.size() != 1 || !valid ||
      valid->elements_.size() != 1) {
    return false;
  }
  // Rank-one INT32 scalar carriers use the dedicated A2/A3 contract.
  // FP32 point-sized partitions also have an ordinary vector DMA route.
  if (tensor->dtype_ == DataType::INT32 && SupportsBufferScalarTransfer(args, load)) return false;
  if (tensor->tensor_view_) {
    const auto& view = *tensor->tensor_view_;
    if (view.layout != ir::TensorLayout::ND || view.pad != ir::PadValue::null) return false;
    if (!view.stride.empty()) {
      const auto stride0 = view.stride.size() == 1 ? ir::As<ir::ConstInt>(view.stride[0]) : nullptr;
      if (!stride0 || stride0->value_ != 1) return false;
    }
  }
  const auto off_type = ir::As<ir::ScalarType>(offsets->elements_[0]->GetType());
  const auto val_type = ir::As<ir::ScalarType>(valid->elements_[0]->GetType());
  if (!off_type || !(off_type->dtype_ == DataType::INDEX || off_type->dtype_.IsSignedInt()) || !val_type ||
      !(val_type->dtype_ == DataType::INDEX || val_type->dtype_.IsSignedInt())) {
    return false;
  }
  const auto extent = ir::As<ir::ConstInt>(valid->elements_[0]);
  if (extent && (extent->value_ <= 0 || extent->value_ > buffer->shape_[1])) return false;
  if (buffer->valid_shape_[1] != -1 && (!extent || extent->value_ != buffer->valid_shape_[1])) return false;
  // Logical vectors use [1, n] at every transfer. A mutable native descriptor
  // must mark both fields dynamic, with the live values checked at the use.
  return buffer->valid_shape_[0] == 1 || (buffer->valid_shape_[0] == -1 && buffer->valid_shape_[1] == -1);
}

bool SupportsLogicalVector1DBufferTransfer(const ir::CallPtr& call) {
  if (!call || SupportsLogicalBufferScalarTransfer(call)) return false;
  const bool load = ir::IsOp(call, "tile.load");
  if (!load && !ir::IsOp(call, "tile.store")) return false;
  if (load ? call->args_.size() != 4 : (call->args_.size() != 3 && call->args_.size() != 4)) return false;
  for (const auto& arg : call->args_)
    if (!arg) return false;
  const auto tile = ir::As<ir::TileType>(load ? call->GetType() : call->args_[0]->GetType());
  const auto tensor = ir::As<ir::TensorType>(call->args_[load ? 0 : 2]->GetType());
  if (!tile || tile->shape_.size() != 1 || !tensor || tensor->shape_.size() != 1) return false;
  const auto physical = PhysicalLogicalVecDescriptor(tile);
  if (!physical) return false;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  const auto& logical_valid = view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  if (logical_valid.size() != 1) return false;
  const auto buf_type = std::make_shared<ir::BufferType>(physical->shape_, tile->dtype_, ir::MemorySpace::Vec,
                                                         MutableBufferValidShape(logical_valid));
  const auto buf_var = std::make_shared<ir::Var>("vec1d", buf_type, call->span_);
  const ir::ExprPtr valid_tuple =
      load ? call->args_[3]
           : std::make_shared<ir::MakeTuple>(std::vector<ir::ExprPtr>{logical_valid[0]}, call->span_);
  if (load) {
    const auto shape_tuple = ir::As<ir::MakeTuple>(call->args_[2]);
    const auto cap = shape_tuple && shape_tuple->elements_.size() == 1
                         ? ir::As<ir::ConstInt>(shape_tuple->elements_[0])
                         : nullptr;
    if (!cap || cap->value_ != physical->shape_[1]) return false;
    return SupportsBufferVector1DTransfer({call->args_[0], call->args_[1], valid_tuple, buf_var}, true);
  }
  return SupportsBufferVector1DTransfer({buf_var, call->args_[1], valid_tuple, call->args_[2]}, false);
}
}  // namespace pypto::backend
