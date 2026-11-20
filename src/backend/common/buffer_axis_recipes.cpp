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

#include "pypto/backend/common/buffer_axis_recipes.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/pto_native_ops.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto::backend {

const std::vector<BufferAxisRecipe>& GetBufferAxisRecipes() {
  static const std::vector<BufferAxisRecipe> recipes = {
      {"tile.row_max", "buffer.row_max", GetPtoNativeOpSpec("tile.row_max").native_op, true},
      {"tile.row_sum", "buffer.row_sum", GetPtoNativeOpSpec("tile.row_sum").native_op, true},
      {"tile.row_min", "buffer.row_min", GetPtoNativeOpSpec("tile.row_min").native_op, true},
      {"tile.row_prod", "buffer.row_prod", GetPtoNativeOpSpec("tile.row_prod").native_op, true},
      {"tile.row_expand_add", "buffer.row_expand_add", GetPtoNativeOpSpec("tile.row_expand_add").native_op,
       false},
      {"tile.row_expand_sub", "buffer.row_expand_sub", GetPtoNativeOpSpec("tile.row_expand_sub").native_op,
       false},
      {"tile.row_expand_mul", "buffer.row_expand_mul", GetPtoNativeOpSpec("tile.row_expand_mul").native_op,
       false},
      {"tile.row_expand_div", "buffer.row_expand_div", GetPtoNativeOpSpec("tile.row_expand_div").native_op,
       false},
      {"tile.row_expand_max", "buffer.row_expand_max", GetPtoNativeOpSpec("tile.row_expand_max").native_op,
       false},
      {"tile.row_expand_min", "buffer.row_expand_min", GetPtoNativeOpSpec("tile.row_expand_min").native_op,
       false},
      {"tile.col_sum", "buffer.col_sum", GetPtoNativeOpSpec("tile.col_sum").native_op, true, 0},
      {"tile.col_max", "buffer.col_max", GetPtoNativeOpSpec("tile.col_max").native_op, true, 0},
      {"tile.col_min", "buffer.col_min", GetPtoNativeOpSpec("tile.col_min").native_op, true, 0},
      {"tile.col_prod", "buffer.col_prod", GetPtoNativeOpSpec("tile.col_prod").native_op, true, 0},
      {"tile.col_expand_add", "buffer.col_expand_add", GetPtoNativeOpSpec("tile.col_expand_add").native_op,
       false, 0},
      {"tile.col_expand_sub", "buffer.col_expand_sub", GetPtoNativeOpSpec("tile.col_expand_sub").native_op,
       false, 0},
      {"tile.col_expand_mul", "buffer.col_expand_mul", GetPtoNativeOpSpec("tile.col_expand_mul").native_op,
       false, 0},
      {"tile.col_expand_div", "buffer.col_expand_div", GetPtoNativeOpSpec("tile.col_expand_div").native_op,
       false, 0},
      {"tile.col_expand_max", "buffer.col_expand_max", GetPtoNativeOpSpec("tile.col_expand_max").native_op,
       false, 0},
      {"tile.col_expand_min", "buffer.col_expand_min", GetPtoNativeOpSpec("tile.col_expand_min").native_op,
       false, 0},
  };
  return recipes;
}

const BufferAxisRecipe* FindLogicalBufferAxisRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferAxisRecipe*> result;
    for (const auto& recipe : GetBufferAxisRecipes()) result.emplace(recipe.logical_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const BufferAxisRecipe* FindBufferAxisRecipe(const std::string& name) {
  static const auto index = [] {
    std::unordered_map<std::string, const BufferAxisRecipe*> result;
    for (const auto& recipe : GetBufferAxisRecipes()) result.emplace(recipe.buffer_op, &recipe);
    return result;
  }();
  const auto found = index.find(name);
  return found == index.end() ? nullptr : found->second;
}

const char* A2A3ColumnBroadcastEncodingError(DataType dtype, int64_t rows, int64_t cols,
                                             int64_t valid_row_bound, int64_t native_valid_cols,
                                             int64_t max_row_stride) {
  // PTO-ISA TColExpandBinOp.hpp selects count mode for a partial or mutable
  // column extent, except on single-row tiles. Count mode loops over rows with
  // unsigned element offsets and sends the valid columns to the count mask.
  if (rows != 1 && native_valid_cols != cols) {
    const auto row_span = std::max(cols, max_row_stride);
    if (row_span <= 0 || valid_row_bound > std::numeric_limits<uint32_t>::max() / row_span) {
      return "requires A2/A3 native count-mode element offsets to fit uint32; split the tile";
    }
    return nullptr;
  }
  constexpr int64_t kMaxRepeatField = std::numeric_limits<uint8_t>::max();
  if (valid_row_bound > kMaxRepeatField) {
    return "requires the A2/A3 native repeat count to fit uint8; split the tile into at most 255 valid rows";
  }
  // The broadcast operand uses zero repeat stride. A single repeat never
  // advances a row, so it can use a wider tile.
  if (valid_row_bound > 1) {
    constexpr int64_t kBlockBits = 32 * 8;
    const auto element_bits = static_cast<int64_t>(dtype.GetBit());
    if (element_bits <= 0 || element_bits > kBlockBits ||
        max_row_stride / (kBlockBits / element_bits) > kMaxRepeatField) {
      return "requires the A2/A3 native repeat stride to fit uint8; split the tile into narrower columns";
    }
  }
  return nullptr;
}

namespace {

bool IsConfiguredForA2A3() {
  return BackendConfig::IsConfigured() &&
         BackendConfig::GetBackend()->GetHandler()->GetPtoTargetArch() == "a2a3";
}

const char* BufferColumnBroadcastEncodingError(const BufferAxisRecipe& recipe,
                                               const ir::BufferTypePtr& destination) {
  if (recipe.axis != 0 || recipe.reduction || !destination || destination->shape_.size() != 2) {
    return nullptr;
  }
  const auto& shape = destination->shape_;
  const auto& valid = destination->valid_shape_;
  return A2A3ColumnBroadcastEncodingError(destination->dtype_, shape[0], shape[1],
                                          valid[0] < 0 ? shape[0] : valid[0], valid[1], shape[1]);
}

bool HasRuntimeMetadata(const std::vector<ir::BufferTypePtr>& types) {
  for (const auto& type : types)
    if (type)
      for (const auto extent : type->valid_shape_)
        if (extent < 0) return true;
  return false;
}

bool NativeElementsFit(const ir::BufferTypePtr& type) {
  return type && type->shape_.size() == 2 && type->shape_[0] > 0 && type->shape_[1] > 0 &&
         static_cast<uint64_t>(type->shape_[0]) <= std::numeric_limits<uint32_t>::max() / type->shape_[1];
}

const char* RuntimeRowEncodingError(const BufferAxisRecipe& recipe,
                                    const std::vector<ir::BufferTypePtr>& types) {
  if (recipe.axis == 0) return nullptr;
  const auto& source = types[0];

  if (!NativeElementsFit(source) || !NativeElementsFit(types[2]))
    return "requires runtime row-operation native uint32 element offsets";
  if (recipe.reduction) {
    // Generic row reduction passes active rows to UB-copy nburst and
    // physical row strides to uint16 reduction operands. Count masks use
    // at most rows*64 elements; the nburst bound also proves no overflow.
    if (source->shape_[0] > std::numeric_limits<uint16_t>::max() ||
        source->shape_[1] / 8 > std::numeric_limits<uint16_t>::max() ||
        types[1]->shape_[1] / 8 > std::numeric_limits<uint16_t>::max() || !NativeElementsFit(types[1]))
      return "requires runtime reduction native uint16 row-count/stride and uint32 scratch offsets";
  } else if (source->shape_[0] >= 256 && source->shape_[1] / 8 > std::numeric_limits<uint8_t>::max()) {
    // Large-row Mode1 selects NormMode; only the small-row path can switch
    // to CountMode when the encoded repeat stride would overflow.
    return "requires runtime row expansion's large-row native uint8 repeat strides";
  }

  return nullptr;
}

// Return a contract diagnostic without throwing: the same predicate must serve
// whole-kernel selection before planning and validation of explicit Buffer IR.
const char* OperandError(const BufferAxisRecipe& recipe, const std::vector<ir::BufferTypePtr>& types) {
  for (size_t i = 0; i < types.size(); ++i) {
    const auto& type = types[i];
    const bool column = recipe.axis == 1 && i == (recipe.reduction ? 2 : 1);
    if (column) {
      if (!IsColumnVectorBufferGeometry(type)) return "requires an aligned contiguous FP32 column vector";
    } else if (!type || type->shape_.size() != 2 || type->dtype_ != DataType::FP32 ||
               type->memory_space_ != ir::MemorySpace::Vec || type->blayout_ != ir::TileLayout::row_major ||
               type->slayout_ != ir::TileLayout::none_box || type->fractal_ != 512 ||
               (type->pad_ != ir::PadValue::null &&
                !((i == 0 || (!recipe.reduction && i == 2)) && IsFullPaddedVec(type))) ||
               type->compact_ != ir::CompactMode::null || type->shape_[1] % 8 != 0) {
      return "requires aligned rank-2 row-major Vec FP32 operands";
    }
    const bool supports_dynamic =
        recipe.axis == 1 || std::string_view(recipe.buffer_op) == "buffer.col_expand_mul";
    const bool metadata_operand =
        supports_dynamic && !(recipe.reduction && i == 1) && !(recipe.axis == 0 && i == 1);
    const bool dynamic_matrix = metadata_operand && type->valid_shape_ == std::vector<int64_t>{-1, -1};
    if (!dynamic_matrix && (type->valid_shape_[0] <= 0 || type->valid_shape_[1] <= 0)) {
      return "requires nonempty static valid extents or qualified broadcast runtime rows";
    }
  }
  const auto& source = types[0];
  if (HasRuntimeMetadata(types)) {
    if (const auto* error = RuntimeRowEncodingError(recipe, types)) return error;
  }
  if (recipe.axis == 0) {
    const auto& row = types[1];
    const bool dynamic_broadcast = std::string_view(recipe.buffer_op) == "buffer.col_expand_mul" &&
                                   (source->valid_shape_ == std::vector<int64_t>{-1, -1} ||
                                    types[2]->valid_shape_ == std::vector<int64_t>{-1, -1});
    if (dynamic_broadcast) {
      if (row->shape_[0] != 1 || row->valid_shape_[0] != 1 || row->shape_[1] != source->shape_[1] ||
          row->valid_shape_[1] != source->shape_[1]) {
        return "requires a single full-valid row preserving physical columns";
      }
      if (types[2]->shape_ != source->shape_) {
        return "requires the broadcast destination to preserve source physical shape";
      }
      for (const auto& matrix : {source, types[2]}) {
        if (matrix->valid_shape_ != std::vector<int64_t>{-1, -1} &&
            matrix->valid_shape_[1] != matrix->shape_[1]) {
          return "requires full-valid columns in the broadcast matrices";
        }
      }
      if (source->valid_shape_[0] >= 0 && types[2]->valid_shape_[0] >= 0 &&
          source->valid_shape_[0] != types[2]->valid_shape_[0]) {
        return "requires full-valid columns and matching broadcast valid rows";
      }
      return nullptr;
    }
    if (row->shape_[0] != 1 || row->valid_shape_[0] != 1 || row->shape_[1] != source->shape_[1] ||
        row->valid_shape_[1] != source->valid_shape_[1]) {
      return "requires a single row preserving physical and valid columns";
    }
    if (!recipe.reduction &&
        (types[2]->shape_ != source->shape_ || types[2]->valid_shape_ != source->valid_shape_)) {
      return "requires the broadcast destination to preserve source physical and valid shape";
    }
    return nullptr;
  }
  const auto& column = types[recipe.reduction ? 2 : 1];
  if (column->shape_[0] != source->shape_[0] ||
      (column->valid_shape_[0] >= 0 && source->valid_shape_[0] >= 0 &&
       column->valid_shape_[0] != source->valid_shape_[0])) {
    return "requires the column vector to preserve physical and valid rows";
  }
  if (recipe.reduction) {
    const auto& workspace = types[1];
    if (workspace->shape_[0] < source->shape_[0] || workspace->shape_[1] < source->shape_[1] ||
        workspace->valid_shape_ != workspace->shape_) {
      return "requires full-valid scratch covering the source physical shape";
    }
  } else if (types[2]->shape_ != source->shape_ || (types[2]->valid_shape_ != std::vector<int64_t>{-1, -1} &&
                                                    source->valid_shape_ != std::vector<int64_t>{-1, -1} &&
                                                    types[2]->valid_shape_ != source->valid_shape_)) {
    return "requires the broadcast destination to preserve source physical and valid shape";
  }
  return nullptr;
}

// out_layout=col_major is the row-reduction recipe's only supported result
// layout, and is also its implicit default when omitted. Reject only kwargs the
// recipe does not support (issue #3319).
bool HasSupportedKwargs(const BufferAxisRecipe& recipe, const ir::CallPtr& call) {
  if (call->kwargs_.empty()) return true;
  if (!(recipe.reduction && recipe.axis == 1)) return false;
  for (const auto& [key, value] : call->kwargs_) {
    if (key != "out_layout" || AnyCast<ir::TileLayout>(value) != ir::TileLayout::col_major) return false;
  }
  return true;
}

bool HasRuntimeBroadcastValid(const ir::TypePtr& type) {
  const auto tile = ir::As<ir::TileType>(type);
  if (!tile) return false;
  const auto view = ir::tile_view_semantics::GetEffectiveTileView(*tile);
  for (const auto& extent : view.valid_shape) {
    if (!ir::As<ir::ConstInt>(extent)) return true;
  }
  return false;
}

bool MatchingLogicalBroadcastValid(const ir::TypePtr& source_type, const ir::TypePtr& result_type) {
  const auto source = ir::As<ir::TileType>(source_type);
  const auto target = ir::As<ir::TileType>(result_type);
  if (!source || !target) return false;
  const auto source_view = ir::tile_view_semantics::GetEffectiveTileView(*source);
  const auto target_view = ir::tile_view_semantics::GetEffectiveTileView(*target);
  const auto& a = source_view.valid_shape.empty() ? source->shape_ : source_view.valid_shape;
  const auto& b = target_view.valid_shape.empty() ? target->shape_ : target_view.valid_shape;
  if (a.size() != 2 || b.size() != 2 || !ir::structural_equal(a[0], b[0])) return false;
  const auto source_columns = ir::As<ir::ConstInt>(source->shape_[1]);
  const auto a_columns = ir::As<ir::ConstInt>(a[1]);
  const auto b_columns = ir::As<ir::ConstInt>(b[1]);
  if (!source_columns || !a_columns || !b_columns || a_columns->value_ != source_columns->value_ ||
      b_columns->value_ != source_columns->value_)
    return false;
  if (const auto rows = ir::As<ir::ConstInt>(a[0])) {
    const auto capacity = ir::As<ir::ConstInt>(source->shape_[0]);
    return capacity && rows->value_ >= 0 && rows->value_ <= capacity->value_;
  }
  const auto scalar = ir::As<ir::ScalarType>(a[0]->GetType());
  return scalar && (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt());
}

bool MatchingLogicalRowValid(const BufferAxisRecipe& recipe, const ir::CallPtr& call) {
  const auto source = ir::As<ir::TileType>(call->args_[0]->GetType());
  const auto target = ir::As<ir::TileType>(call->GetType());
  if (!source || !target) return false;
  const auto a = ir::tile_view_semantics::GetEffectiveTileView(*source);
  const auto b = ir::tile_view_semantics::GetEffectiveTileView(*target);
  const auto& av = a.valid_shape.empty() ? source->shape_ : a.valid_shape;
  const auto& bv = b.valid_shape.empty() ? target->shape_ : b.valid_shape;
  if (av.size() != 2 || bv.size() != 2 || !ir::structural_equal(av[0], bv[0])) return false;
  const auto columns = ir::As<ir::ConstInt>(source->shape_[1]);
  const auto ac = ir::As<ir::ConstInt>(av[1]);
  const auto bc = ir::As<ir::ConstInt>(bv[1]);
  if (!columns || !ac || ac->value_ != columns->value_ || !bc ||
      bc->value_ != (recipe.reduction ? 1 : columns->value_))
    return false;
  if (!recipe.reduction) {
    const auto carrier = ir::As<ir::TileType>(call->args_[1]->GetType());
    if (!carrier) return false;
    const auto c = ir::tile_view_semantics::GetEffectiveTileView(*carrier);
    const auto& cv = c.valid_shape.empty() ? carrier->shape_ : c.valid_shape;
    const auto cc = cv.size() == 2 ? ir::As<ir::ConstInt>(cv[1]) : nullptr;
    if (!cc || cc->value_ != 1 || !ir::structural_equal(av[0], cv[0])) return false;
  }
  return true;
}

}  // namespace

bool SupportsLogicalBufferAxis(const BufferAxisRecipe& recipe, const ir::CallPtr& call) {
  // Binary-tree col_sum and optional-scratch/packed row_expand_add have
  // separate storage contracts and remain on Tile.
  if (call->args_.size() != recipe.InputCount() || !HasSupportedKwargs(recipe, call)) return false;
  std::vector<ir::BufferTypePtr> types;
  if (std::string_view(recipe.buffer_op) == "buffer.col_expand_mul" &&
      (HasRuntimeBroadcastValid(call->args_[0]->GetType()) || HasRuntimeBroadcastValid(call->GetType()))) {
    if (!MatchingLogicalBroadcastValid(call->args_[0]->GetType(), call->GetType())) return false;
    types = {PhysicalLogicalVecDescriptor(call->args_[0]->GetType()),
             StaticLogicalBufferDescriptor(call->args_[1]->GetType()),
             PhysicalLogicalVecDescriptor(call->GetType())};
    // These projections describe capacity. Both matrices have the same
    // symbolic valid rows, so lowering gives them mutable native validity.
    if (OperandError(recipe, types) != nullptr) {
      return false;
    }
    const auto& dst = types[2];
    return A2A3ColumnBroadcastEncodingError(dst->dtype_, dst->shape_[0], dst->shape_[1], dst->shape_[0],
                                            /*native_valid_cols=*/-1, dst->shape_[1]) == nullptr;
  } else {
    for (const auto& arg : call->args_) types.push_back(StaticLogicalBufferDescriptor(arg->GetType(), true));
    types.push_back(StaticLogicalBufferDescriptor(call->GetType(), true));
    if (OperandError(recipe, types) == nullptr) {
      // Whole-function Buffer admission is A2/A3-only. Ordinary Functional
      // allocations use mutable native validity and may safely run in count mode.
      return recipe.axis != 0 || recipe.reduction || !IsConfiguredForA2A3() ||
             BufferColumnBroadcastEncodingError(recipe, types.back()) == nullptr;
    }
    if (recipe.axis != 1 || !MatchingLogicalRowValid(recipe, call)) return false;
    types = {PhysicalLogicalVecDescriptor(call->args_[0]->GetType()),
             recipe.reduction ? StaticLogicalBufferDescriptor(call->args_[1]->GetType())
                              : PhysicalLogicalVecDescriptor(call->args_[1]->GetType()),
             PhysicalLogicalVecDescriptor(call->GetType())};
  }
  return OperandError(recipe, types) == nullptr && RuntimeRowEncodingError(recipe, types) == nullptr;
}

void ValidateBufferAxisOperands(const BufferAxisRecipe& recipe, const std::vector<ir::ExprPtr>& args) {
  CHECK(args.size() == recipe.InputCount() + 1)
      << recipe.buffer_op
      << (recipe.InputCount() == 2 ? " requires two inputs and an explicit destination"
                                   : " requires one input and an explicit destination");
  std::vector<ir::BufferTypePtr> types;
  for (const auto& arg : args) {
    CHECK(arg) << recipe.buffer_op << " operands must not be null";
    types.push_back(ir::As<ir::BufferType>(arg->GetType()));
  }
  const auto* error = OperandError(recipe, types);
  CHECK(!error) << recipe.buffer_op << " " << error;
  if (recipe.axis == 0 && !recipe.reduction && IsConfiguredForA2A3()) {
    const auto* encoding_error = BufferColumnBroadcastEncodingError(recipe, types.back());
    CHECK(!encoding_error) << recipe.buffer_op << " " << encoding_error;
  }
}

void ValidateA2A3BufferAxisEncoding(const BufferAxisRecipe& recipe, const ir::CallPtr& call) {
  if (recipe.axis == 0 && !recipe.reduction) {
    INTERNAL_CHECK_SPAN(call->args_.size() == 3, call->span_)
        << "Internal error: expected source, broadcast row and destination for " << recipe.buffer_op;
    const auto destination = ir::As<ir::BufferType>(call->args_.back()->GetType());
    const auto* error = BufferColumnBroadcastEncodingError(recipe, destination);
    CHECK_SPAN(!error, call->span_) << recipe.buffer_op << " " << error;
  }
}

}  // namespace pypto::backend
