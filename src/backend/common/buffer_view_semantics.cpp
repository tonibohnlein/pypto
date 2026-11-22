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

#include "pypto/backend/common/buffer_view_semantics.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/const_fold.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

std::optional<uint64_t> DenseBufferBytes(const ir::BufferTypePtr& type) {
  if (!type || type->blayout_ != ir::TileLayout::row_major || type->slayout_ != ir::TileLayout::none_box ||
      type->fractal_ != 512 || type->pad_ != ir::PadValue::null || type->compact_ != ir::CompactMode::null) {
    return std::nullopt;
  }
  return PhysicalBufferBytes(type);
}

std::optional<uint64_t> PhysicalBufferBytes(const ir::BufferTypePtr& type) {
  if (!type || (type->shape_.size() != 1 && type->shape_.size() != 2) || type->dtype_.GetBit() == 0 ||
      type->dtype_.GetBit() % 8 != 0) {
    return std::nullopt;
  }
  uint64_t bytes = type->dtype_.GetBit() / 8;
  for (const auto extent : type->shape_) {
    if (extent <= 0 || static_cast<uint64_t>(extent) > std::numeric_limits<uint64_t>::max() / bytes) {
      return std::nullopt;
    }
    bytes *= static_cast<uint64_t>(extent);
  }
  return bytes;
}

bool SupportsA2A3VecCopyDescriptor(const ir::BufferTypePtr& type) {
  const auto bytes = PhysicalBufferBytes(type);
  constexpr uint64_t kBlockBytes = 32;
  return bytes && *bytes / kBlockBytes <= std::numeric_limits<uint16_t>::max() &&
         ((IsAlignedDenseVecBufferGeometry(type) && type->shape_ == type->valid_shape_) ||
          IsMinimumPaddedVecRowGeometry(type) ||
          (IsColumnVectorBufferGeometry(type) && type->valid_shape_[0] != 0 &&
           (type->valid_shape_[1] == -1 || type->valid_shape_[1] == 1)));
}

bool SupportsLogicalBufferVecCopy(const ir::CallPtr& call) {
  if (call->args_.size() != 1) {
    return false;
  }
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto target = ir::As<ir::TileType>(call->GetType());
  if (!source || !target) {
    return false;
  }
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*target);
  if (view != ir::tile_view_semantics::GetEffectiveTileView(*source)) {
    return false;
  }
  const bool row_prefix = IsMinimumPaddedVecRowPrefix(source, view);
  const bool contiguous_column = IsColumnVectorBufferTile(source, view);
  const auto a = (row_prefix || contiguous_column)
                     ? PhysicalLogicalVecDescriptor(source, /*allow_full_pad=*/false, row_prefix)
                     : StaticLogicalBufferDescriptor(source);
  const auto b = (row_prefix || contiguous_column)
                     ? PhysicalLogicalVecDescriptor(target, /*allow_full_pad=*/false, row_prefix)
                     : StaticLogicalBufferDescriptor(target);
  if (contiguous_column) {
    const auto& valid = view.valid_shape.empty() ? source->shape_ : view.valid_shape;
    const auto columns = valid.size() == 2 ? ir::As<ir::ConstInt>(valid[1]) : nullptr;
    if (!columns || columns->value_ != 1) {
      return false;
    }
  }
  if (!SupportsA2A3VecCopyDescriptor(a) || !ir::structural_equal(a, b)) {
    return false;
  }
  for (const auto& [name, value] : call->kwargs_) {
    const bool valid =
        (name == "target_memory" && call->GetKwarg<ir::MemorySpace>(name) == ir::MemorySpace::Vec) ||
        (name == "blayout" && call->GetKwarg<ir::TileLayout>(name) == a->blayout_) ||
        (name == "slayout" && call->GetKwarg<ir::TileLayout>(name) == a->slayout_);
    if (!valid) {
      return false;
    }
  }
  return true;
}

bool SupportsA2A3VecInsert(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination,
                           const ir::ExprPtr& row, const ir::ExprPtr& column) {
  constexpr int64_t kBlockBytes = 32;
  // A2/A3 block DMA uses a 12-bit burst count and 16-bit lengths/gaps.
  // The unaligned tail uses vector repeat strides measured in 32-byte blocks.
  constexpr int64_t kMaxBurstCount = 4095;
  constexpr int64_t kMaxRepeatStride = 255;
  constexpr int64_t kMaxU16 = std::numeric_limits<uint16_t>::max();
  const auto src_bytes = DenseBufferBytes(source);
  if (!src_bytes || !DenseBufferBytes(destination) || source->memory_space_ != ir::MemorySpace::Vec ||
      destination->memory_space_ != ir::MemorySpace::Vec || source->shape_.size() != 2 ||
      destination->shape_.size() != 2 || source->dtype_ != destination->dtype_ ||
      destination->valid_shape_ != destination->shape_) {
    return false;
  }
  const auto dtype = source->dtype_;
  if (dtype != DataType::FP16 && dtype != DataType::BF16 && dtype != DataType::FP32) {
    return false;
  }
  const auto per_block = kBlockBytes / dtype.GetByte();
  const auto row_offset = ir::As<ir::ConstInt>(row);
  const auto col_offset = ir::As<ir::ConstInt>(column);
  return row_offset && col_offset && row_offset->value_ >= 0 && col_offset->value_ >= 0 &&
         ir::arith::detail::ConstIntFitsDtype(row_offset->value_, row_offset->dtype()) &&
         ir::arith::detail::ConstIntFitsDtype(col_offset->value_, col_offset->dtype()) &&
         row_offset->value_ <= kMaxU16 && col_offset->value_ <= kMaxU16 &&
         row_offset->value_ <= destination->shape_[0] - source->shape_[0] &&
         col_offset->value_ <= destination->shape_[1] - source->shape_[1] &&
         col_offset->value_ % per_block == 0 && source->shape_[1] % per_block == 0 &&
         destination->shape_[1] % per_block == 0 && source->shape_[0] <= kMaxBurstCount &&
         destination->shape_[0] <= kMaxU16 && source->shape_[1] <= kMaxRepeatStride * per_block &&
         destination->shape_[1] <= kMaxRepeatStride * per_block && *src_bytes / kBlockBytes <= kMaxU16 &&
         source->valid_shape_[0] > 0 && source->valid_shape_[0] <= source->shape_[0] &&
         source->valid_shape_[1] > 0 && source->valid_shape_[1] <= source->shape_[1] &&
         (source->valid_shape_[0] != 1 || source->valid_shape_[1] != 1);
}

std::optional<PeerTransferBuffer> DescribePeerTransferBuffer(const ir::TileTypePtr& tile) {
  if (!tile || tile->shape_.size() != 2) return std::nullopt;
  const auto space = tile->GetMemorySpace();
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  const auto rows = ir::As<ir::ConstInt>(tile->shape_[0]);
  const auto cols = ir::As<ir::ConstInt>(tile->shape_[1]);
  const auto offset = ir::As<ir::ConstInt>(view.start_offset);
  if (!rows || !cols || rows->value_ <= 0 || cols->value_ <= 0 || view.valid_shape.size() != 2 ||
      (view.start_offset && (!offset || offset->value_ != 0)) || view.compact != ir::CompactMode::null) {
    return std::nullopt;
  }
  if (space == ir::MemorySpace::Vec) {
    if (!IsDenseBufferTransferDtype(tile->dtype_) || !IsDenseVecBufferTile(tile, view, view.pad) ||
        cols->value_ % (32 / static_cast<int64_t>(tile->dtype_.GetByte())) != 0) {
      return std::nullopt;
    }
    if (!view.stride.empty()) {
      if (view.stride.size() != 2) return std::nullopt;
      const auto pitch = ir::As<ir::ConstInt>(view.stride[0]);
      const auto unit = ir::As<ir::ConstInt>(view.stride[1]);
      if (!pitch || !unit || pitch->value_ != cols->value_ || unit->value_ != 1) return std::nullopt;
    }
  } else {
    // The byte product is a physical capacity only for complete native boxes.
    // INT32 accumulators have a target-specific L0C M pitch; keep those out
    // until the selected transport accounts for that additional alignment.
    if ((space != ir::MemorySpace::Acc && space != ir::MemorySpace::Mat) || !view.stride.empty() ||
        (space == ir::MemorySpace::Acc && tile->dtype_ != DataType::FP32) ||
        !IsMatrixBufferDtype(*space, tile->dtype_) || view.blayout != ir::TileLayout::col_major ||
        view.slayout != ir::TileLayout::row_major ||
        view.fractal != ir::tile_view_semantics::GetImplicitFractal(space)) {
      return std::nullopt;
    }
    const auto box = ir::tile_view_semantics::GetBoxedTileAlignment(view, tile->dtype_);
    if (!box || rows->value_ % box->rows != 0 || cols->value_ % box->cols != 0) return std::nullopt;
  }
  const std::vector<int64_t> shape{rows->value_, cols->value_};
  std::vector<int64_t> valid;
  for (size_t axis = 0; axis < shape.size(); ++axis) {
    const auto& extent = view.valid_shape[axis];
    const auto type = extent ? ir::As<ir::ScalarType>(extent->GetType()) : nullptr;
    if (!type || (!type->dtype_.IsInt() && type->dtype_ != DataType::INDEX)) return std::nullopt;
    const auto constant = ir::As<ir::ConstInt>(extent);
    if (constant && (constant->value_ < 0 || constant->value_ > shape[axis])) return std::nullopt;
    valid.push_back(constant ? constant->value_ : -1);
  }
  auto descriptor = std::make_shared<ir::BufferType>(shape, tile->dtype_, *space, valid, view.blayout,
                                                     view.slayout, view.fractal, view.pad, view.compact);
  const auto bytes = PhysicalBufferBytes(descriptor);
  if (!bytes || *bytes > std::numeric_limits<int64_t>::max()) return std::nullopt;
  return PeerTransferBuffer{std::move(descriptor), static_cast<int64_t>(*bytes)};
}

bool IsContiguousVectorReshape(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination) {
  if (!source || !destination || source->valid_shape_ != source->shape_ ||
      destination->valid_shape_ != destination->shape_)
    return false;
  const auto column = IsColumnVectorBuffer(source) ? source : destination;
  const auto row = column == source ? destination : source;
  const auto bytes = PhysicalBufferBytes(source);
  return bytes && IsColumnVectorBuffer(column) && IsAlignedDenseVecBuffer(row) &&
         row->dtype_ == DataType::FP32 && row->shape_[0] == 1 && column->shape_[0] > 0 &&
         row->shape_[1] == column->shape_[0] && bytes == PhysicalBufferBytes(destination);
}

std::optional<int64_t> LogicalContiguousVectorLength(const ir::TileTypePtr& type) {
  if (!type || type->shape_.size() != 2 || type->dtype_ != DataType::FP32 ||
      type->GetMemorySpace() != ir::MemorySpace::Vec)
    return std::nullopt;
  const auto rows = ir::As<ir::ConstInt>(type->shape_[0]);
  const auto columns = ir::As<ir::ConstInt>(type->shape_[1]);
  if (!rows || !columns) return std::nullopt;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*type);
  const bool column = IsColumnVectorBufferTile(type, view);
  const bool row = IsDenseVecBufferTile(type, view) && rows->value_ == 1 && columns->value_ > 0 &&
                   columns->value_ % 8 == 0;
  if (!column && !row) return std::nullopt;
  if (view.start_offset) {
    const auto offset = ir::As<ir::ConstInt>(view.start_offset);
    if (!offset || offset->value_ != 0) return std::nullopt;
  }
  if (!view.stride.empty()) {
    if (view.stride.size() != 2) return std::nullopt;
    for (size_t axis = 0; axis < 2; ++axis) {
      const auto stride = ir::As<ir::ConstInt>(view.stride[axis]);
      const auto expected = column ? (axis ? rows->value_ : 1) : (axis ? 1 : columns->value_);
      if (!stride || stride->value_ != expected) return std::nullopt;
    }
  }
  return column ? rows->value_ : columns->value_;
}

bool SupportsLogicalContiguousVectorReshape(const ir::CallPtr& call) {
  if (!call || !ir::IsOp(call, "tile.reshape") || call->args_.size() != 2 || !call->kwargs_.empty())
    return false;
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto target = ir::As<ir::TileType>(call->GetType());
  const auto requested = ir::As<ir::MakeTuple>(call->args_[1]);
  if (!target || !requested || requested->elements_.size() != target->shape_.size()) return false;
  for (size_t axis = 0; axis < target->shape_.size(); ++axis) {
    const auto dimension = ir::As<ir::ConstInt>(requested->elements_[axis]);
    const auto declared = ir::As<ir::ConstInt>(target->shape_[axis]);
    if (!dimension || !declared || dimension->value_ != declared->value_) return false;
  }
  const auto length = LogicalContiguousVectorLength(source);
  if (!length || LogicalContiguousVectorLength(target) != length) return false;
  const auto a = ir::tile_view_semantics::GetEffectiveTileView(*source);
  const auto b = ir::tile_view_semantics::GetEffectiveTileView(*target);
  if (a.blayout == b.blayout) {
    // An identity is still the existing qualified contiguous vector, not an
    // arbitrary same-byte rectangle. Preserve every physical axis and its
    // actual logical valid extent; stride/zero-offset proofs come from length.
    if (a.slayout != b.slayout || a.fractal != b.fractal || a.pad != b.pad || a.compact != b.compact)
      return false;
    for (size_t axis = 0; axis < source->shape_.size(); ++axis)
      if (!ir::structural_equal(source->shape_[axis], target->shape_[axis])) return false;
    const auto& av = a.valid_shape.empty() ? source->shape_ : a.valid_shape;
    const auto& bv = b.valid_shape.empty() ? target->shape_ : b.valid_shape;
    if (av.size() != 2 || bv.size() != 2) return false;
    const size_t active_axis = a.blayout == ir::TileLayout::col_major ? 0 : 1;
    const auto fixed = ir::As<ir::ConstInt>(av[1 - active_axis]);
    return fixed && fixed->value_ == 1 && ir::structural_equal(av[0], bv[0]) &&
           ir::structural_equal(av[1], bv[1]);
  }
  const auto& av = a.valid_shape.empty() ? source->shape_ : a.valid_shape;
  const auto& bv = b.valid_shape.empty() ? target->shape_ : b.valid_shape;
  if (av.size() != 2 || bv.size() != 2) return false;
  const size_t source_axis = a.blayout == ir::TileLayout::col_major ? 0 : 1;
  const size_t target_axis = b.blayout == ir::TileLayout::col_major ? 0 : 1;
  const auto fixed_a = ir::As<ir::ConstInt>(av[1 - source_axis]);
  const auto fixed_b = ir::As<ir::ConstInt>(bv[1 - target_axis]);
  return fixed_a && fixed_a->value_ == 1 && fixed_b && fixed_b->value_ == 1 &&
         ir::structural_equal(av[source_axis], bv[target_axis]);
}

namespace {

ir::BufferTypePtr StaticViewDescriptor(const ir::TypePtr& type, const std::string& name) {
  auto buffer = ir::As<ir::BufferType>(type);
  const auto bytes = DenseBufferBytes(buffer);
  CHECK(buffer && buffer->shape_.size() == 2 && buffer->memory_space_ == ir::MemorySpace::Vec &&
        (IsDenseBufferTransferDtype(buffer->dtype_) || buffer->dtype_ == DataType::INT16 ||
         (buffer->dtype_ == DataType::UINT8 || buffer->dtype_ == DataType::UINT32)) &&
        bytes.has_value())
      << name << " requires dense rank-2 row-major Vec FP16/BF16/FP32/INT16/INT32/UINT8/UINT32 descriptors";
  for (const auto extent : buffer->valid_shape_) {
    CHECK(extent >= 0) << name << " requires static valid extents";
  }
  const auto row_bytes = *bytes / static_cast<uint64_t>(buffer->shape_[0]);
  CHECK(row_bytes % 32 == 0) << name << " requires physical rows aligned to 32 bytes";
  return buffer;
}

// A fractal window is never sliced or re-pitched; the only static alias is a
// relabel of the whole window in the same space (for example NZ <-> ZN).
void ValidateMatrixReshape(const ir::BufferTypePtr& source, const ir::BufferTypePtr& destination) {
  CHECK(destination && destination->memory_space_ == source->memory_space_)
      << "buffer.reshape of a " << ir::MemorySpaceToString(source->memory_space_)
      << " buffer must stay in the same memory space";
  CHECK(source->shape_.size() == 2 && destination->shape_.size() == 2 &&
        destination->dtype_ == source->dtype_)
      << "buffer.reshape of a matrix-space buffer requires rank-2 descriptors with the same element type";
  const auto bytes = PhysicalBufferBytes(source);
  CHECK(bytes && bytes == PhysicalBufferBytes(destination))
      << "buffer.reshape requires equal physical byte sizes";
  for (const auto& type : {source, destination}) {
    for (const auto extent : type->valid_shape_) {
      CHECK(extent >= 0) << "buffer.reshape of a matrix-space buffer requires static valid extents";
    }
  }
}

}  // namespace

// A strided window of a dense row-major Vec buffer: the result keeps the
// source's row pitch, so any static-shape window fits, at static or runtime
// offsets. Static byte windows of the UINT8[N,32] storage root are the
// full-width special case. A result valid dimension is either static in the
// descriptor or dynamic (-1); the optional valid tuple states every dimension
// and is required when any is dynamic or smaller than the physical shape.
void ValidateBufferSubview(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK((args.size() == 2 || args.size() == 3) && args[0] && args[1])
      << "Internal error: buffer.subview requires a source, an offsets tuple and optional valid extents";
  auto source = ir::As<ir::BufferType>(args[0]->GetType());
  auto destination = ir::As<ir::BufferType>(result);
  for (const auto& [type, role] : {std::pair{source, "source"}, std::pair{destination, "result"}}) {
    CHECK(type && type->shape_.size() == 2 && type->memory_space_ == ir::MemorySpace::Vec &&
          DenseBufferBytes(type).has_value())
        << "buffer.subview " << role << " must be a dense rank-2 row-major Vec buffer";
  }
  CHECK(source->dtype_ == destination->dtype_)
      << "buffer.subview must keep the element type, got " << source->dtype_.ToString() << " -> "
      << destination->dtype_.ToString();
  auto offsets = ir::As<ir::MakeTuple>(args[1]);
  CHECK(offsets && offsets->elements_.size() == 2) << "buffer.subview offsets must be a rank-2 MakeTuple";
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto& offset = offsets->elements_[axis];
    auto scalar = offset ? ir::As<ir::ScalarType>(offset->GetType()) : nullptr;
    CHECK(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX))
        << "buffer.subview offset " << axis << " must be an integer or INDEX scalar";
    const auto extent = destination->shape_[axis];
    CHECK(extent <= source->shape_[axis]) << "buffer.subview window dimension " << axis << " (" << extent
                                          << ") exceeds the source (" << source->shape_[axis] << ")";
    if (auto constant = ir::As<ir::ConstInt>(offset)) {
      CHECK(constant->value_ >= 0 && constant->value_ <= source->shape_[axis] - extent)
          << "buffer.subview window exceeds source capacity on dimension " << axis;
    }
  }
  const bool dynamic = destination->valid_shape_[0] < 0 || destination->valid_shape_[1] < 0;
  CHECK((!dynamic && destination->valid_shape_ == destination->shape_) || args.size() == 3)
      << "buffer.subview requires a valid-extents tuple when its result has dynamic or narrowed valid "
         "dimensions";
  if (args.size() == 3) {
    auto valid = ir::As<ir::MakeTuple>(args[2]);
    CHECK(valid && valid->elements_.size() == 2) << "buffer.subview valid extents must be a rank-2 MakeTuple";
    for (size_t axis = 0; axis < 2; ++axis) {
      const auto& extent = valid->elements_[axis];
      auto scalar = extent ? ir::As<ir::ScalarType>(extent->GetType()) : nullptr;
      CHECK(scalar && (scalar->dtype_.IsInt() || scalar->dtype_ == DataType::INDEX))
          << "buffer.subview valid extent " << axis << " must be an integer or INDEX scalar";
      auto constant = ir::As<ir::ConstInt>(extent);
      if (constant) {
        CHECK(constant->value_ > 0 && constant->value_ <= destination->shape_[axis])
            << "buffer.subview valid extent " << axis << " must be positive and no greater than "
            << destination->shape_[axis];
      }
      if (destination->valid_shape_[axis] >= 0) {
        CHECK(constant && constant->value_ == destination->valid_shape_[axis])
            << "buffer.subview valid extent " << axis << " must equal the static descriptor dimension";
      } else {
        CHECK(!constant) << "buffer.subview valid extent " << axis
                         << " must be nonconstant for a dynamic descriptor dimension";
      }
    }
  }
}

void ValidateBufferReshape(const std::vector<ir::ExprPtr>& args, const ir::TypePtr& result) {
  INTERNAL_CHECK(args.size() == 1 && args[0]) << "Internal error: buffer.reshape requires exactly one source";
  auto matrix_source = ir::As<ir::BufferType>(args[0]->GetType());
  const auto destination_type = ir::As<ir::BufferType>(result);
  if (IsContiguousVectorReshape(matrix_source, destination_type)) return;
  if (IsColumnVectorBuffer(matrix_source) || IsColumnVectorBuffer(destination_type)) {
    CHECK(IsColumnVectorBuffer(matrix_source) && IsColumnVectorBuffer(destination_type) &&
          matrix_source->shape_ == destination_type->shape_)
        << "buffer.reshape of a column vector must preserve its physical descriptor";
    return;
  }
  if (matrix_source && IsMatrixBufferSpace(matrix_source->memory_space_)) {
    ValidateMatrixReshape(matrix_source, ir::As<ir::BufferType>(result));
    return;
  }
  auto source = StaticViewDescriptor(args[0]->GetType(), "buffer.reshape");
  auto destination = StaticViewDescriptor(result, "buffer.reshape");
  CHECK(DenseBufferBytes(source) == DenseBufferBytes(destination))
      << "buffer.reshape requires equal physical byte sizes";
}

}  // namespace pypto::backend
