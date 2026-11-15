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

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "memory_reuse/internal.h"
#include "memory_reuse/storage.h"
#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_axis_recipes.h"
#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_transpose_recipe.h"
#include "pypto/backend/common/buffer_type_support.h"
#include "pypto/backend/common/buffer_view_semantics.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/tile_view_semantics.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/buffer_indexed_analysis.h"
#include "pypto/ir/transforms/utils/buffer_tuple_results.h"
#include "pypto/ir/transforms/utils/buffer_valid_analysis.h"
#include "pypto/ir/transforms/utils/buffer_view_analysis.h"
#include "pypto/ir/transforms/utils/loop_step_qualification.h"
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/slot_index_qualification.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"
#include "src/ir/transforms/buffer_ir_admission.h"

namespace pypto {
namespace ir {
namespace {

// Capability is decided before any Buffer-specific storage mutation. This is
// deliberately narrower than the explicit low-level lowering API. No exception
// is interpreted as an unsupported capability: malformed selected IR must fail.
class BufferKernelSupport : public IRVisitor {
 public:
  explicit BufferKernelSupport(const FunctionPtr& function)
      : body_(function->body_),
        return_types_(function->return_types_),
        views_(function->body_),
        indexed_(function->body_),
        tuples_(function->body_) {
    for (size_t i = 0; i < function->params_.size(); ++i) {
      const auto& param = function->params_[i];
      if (As<TensorType>(param->GetType())) {
        tensor_aliases_.emplace(param.get(), param);
        directions_.emplace(param.get(), function->param_directions_[i]);
      }
    }
  }

  BufferAdmission admission;
  bool has_load = false;
  bool has_fill = false;
  bool has_store = false;

  static bool ConstantEquals(const ExprPtr& value, int64_t expected) {
    const auto constant = As<ConstInt>(value);
    return constant && constant->value_ == expected;
  }

  static bool SameShape(const std::vector<ExprPtr>& lhs, const std::vector<ExprPtr>& rhs) {
    return lhs.size() == rhs.size() &&
           std::equal(lhs.begin(), lhs.end(), rhs.begin(),
                      [](const ExprPtr& a, const ExprPtr& b) { return structural_equal(a, b); });
  }

  static bool StaticShape(const std::vector<ExprPtr>& shape) {
    return shape.size() == 2 && As<ConstInt>(shape[0]) && As<ConstInt>(shape[1]) &&
           As<ConstInt>(shape[0])->value_ > 0 && As<ConstInt>(shape[1])->value_ > 1;
  }

  void CheckType(const TypePtr& type) {
    if (!checked_types_.insert(type.get()).second) return;
    if (auto tile = As<TileType>(type)) {
      const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
      const bool column = backend::IsColumnVectorBufferTile(tile, view);
      if (!column && !StaticShape(tile->shape_)) {
        admission.Require(false, BufferAdmissionProperty::TileShape);
        return;
      }
      const auto cols = As<ConstInt>(tile->shape_[1])->value_;
      const auto space = tile->GetMemorySpace();
      if (space && backend::IsMatrixBufferSpace(*space)) {
        const auto box = tile_view_semantics::GetBoxedTileAlignment(view, tile->dtype_);
        admission.Require(
            backend::IsMatrixBufferDtype(*space, tile->dtype_) && box &&
                (view.fractal == (*space == MemorySpace::Acc ? 1024 : 512)) &&
                (view.blayout == TileLayout::row_major || view.blayout == TileLayout::col_major) &&
                view.pad == PadValue::null && view.compact == CompactMode::null && view.stride.empty(),
            BufferAdmissionProperty::TypeLayout);
        if (box)
          admission.Require(As<ConstInt>(tile->shape_[0])->value_ % box->rows == 0 && cols % box->cols == 0,
                            BufferAdmissionProperty::TypeLayout);
      } else if (column) {
        const auto rows = As<ConstInt>(tile->shape_[0])->value_;
        admission.Require(
            view.stride.empty() || (view.stride.size() == 2 && ConstantEquals(view.stride[0], 1) &&
                                    ConstantEquals(view.stride[1], rows)),
            BufferAdmissionProperty::TypeLayout);
      } else {
        admission.Require(
            backend::IsDenseVecBufferTile(tile, view) && cols % (256 / tile->dtype_.GetBit()) == 0,
            BufferAdmissionProperty::TypeLayout);
        admission.Require(
            view.stride.empty() || (view.stride.size() == 2 && ConstantEquals(view.stride[0], cols) &&
                                    ConstantEquals(view.stride[1], 1)),
            BufferAdmissionProperty::TypeLayout);
      }
      admission.Require(!view.start_offset || ConstantEquals(view.start_offset, 0),
                        BufferAdmissionProperty::TypeLayout);
      if (!view.valid_shape.empty()) {
        admission.Require(view.valid_shape.size() == tile->shape_.size(),
                          BufferAdmissionProperty::TypeLayout);
        for (size_t i = 0; admission.supported() && i < view.valid_shape.size(); ++i) {
          const auto valid = As<ConstInt>(view.valid_shape[i]);
          const auto scalar = As<ScalarType>(view.valid_shape[i]->GetType());
          admission.Require(valid
                                ? valid->value_ >= 0 && valid->value_ <= As<ConstInt>(tile->shape_[i])->value_
                                : space == MemorySpace::Vec && scalar &&
                                      (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt()),
                            BufferAdmissionProperty::TypeLayout);
          if (space != MemorySpace::Vec || column)
            admission.Require(valid && valid->value_ > 0, BufferAdmissionProperty::TypeLayout);
        }
      }
      if (const auto memory = tile->memref_.value_or(nullptr)) {
        admission.Require(!memory->is_pinned_, BufferAdmissionProperty::TypeLayout);
        if (memory->slot_count_ > 1 || memory->slot_index_)
          admission.Require(QualifySlot(tile, memory), BufferAdmissionProperty::SlotGeometry);
      }
    } else if (auto tensor = As<TensorType>(type)) {
      admission.Require(
          backend::IsBufferTensorShape(tensor->shape_) &&
              (backend::IsDenseBufferTransferDtype(tensor->dtype_) || tensor->dtype_ == DataType::INT8),
          BufferAdmissionProperty::TypeLayout);
      if (admission.supported() && tensor->tensor_view_) {
        const auto& view = *tensor->tensor_view_;
        admission.Require(view.layout == TensorLayout::ND && view.pad == PadValue::null,
                          BufferAdmissionProperty::TypeLayout);
        admission.Require(view.valid_shape.empty() || SameShape(view.valid_shape, tensor->shape_),
                          BufferAdmissionProperty::TypeLayout);
        admission.Require(
            view.stride.empty() || (view.stride.size() == 2 &&
                                    ConstantEquals(view.stride[0], As<ConstInt>(tensor->shape_[1])->value_) &&
                                    ConstantEquals(view.stride[1], 1)),
            BufferAdmissionProperty::TypeLayout);
      }
    } else if (!As<ScalarType>(type) && !As<PtrType>(type) && !As<VoidType>(type) && !As<TupleType>(type)) {
      admission.Require(false, BufferAdmissionProperty::TypeLayout);
    }
  }

  void VisitExpr(const ExprPtr& expr) override {
    if (!admission.supported() || !expr) return;
    auto context = admission.At(expr.get());
    if (As<TupleGetItemExpr>(expr) || As<Submit>(expr) || As<Pow>(expr)) {
      admission.Require(false, BufferAdmissionProperty::ExpressionForm);
      return;
    }
    const auto physical_result = assigned_slot_types_.find(expr.get());
    CheckType(physical_result == assigned_slot_types_.end() ? expr->GetType() : physical_result->second);
    if (const auto var = AsVarLike(expr)) {
      if (As<TupleType>(var->GetType())) supported = false;
      if (const auto tile = As<TileType>(var->GetType())) {
        if (const auto memory = tile->memref_.value_or(nullptr)) {
          const auto offset = As<ConstInt>(memory->byte_offset_);
          const auto view = views_.Get(var);
          // Only a slice with a known typed parent can carry a symbolic
          // member offset. Allocation origins themselves remain static.
          if (memory->slot_count_ > 1 || memory->slot_index_) {
            admission.Require(!view.window, BufferAdmissionProperty::ExpressionForm);
            if (admission.supported() && !loop_slots_.empty()) {
              const auto index = *memory->slot_index_;
              auto [entry, inserted] = loop_slots_.front().emplace(memory->base_.get(), index);
              admission.Require(inserted || structural_equal(entry->second, index),
                                BufferAdmissionProperty::ExpressionForm);
            }
          } else {
            admission.Require(offset ? offset->value_ >= 0 && offset->value_ % 32 == 0
                                     : view.window && view.root && view.row_bytes.has_value(),
                              BufferAdmissionProperty::ExpressionForm);
          }
        }
      }
    }
    IRVisitor::VisitExpr(expr);
  }

  void VisitStmt(const StmtPtr& stmt) override {
    if (!admission.supported()) return;
    const auto assign = As<AssignStmt>(stmt);
    const auto call = assign ? As<Call>(assign->value_) : nullptr;
    auto context = admission.At(call ? static_cast<const IRNode*>(call.get()) : stmt.get());
    if (!As<SeqStmts>(stmt) && !As<AssignStmt>(stmt) && !As<EvalStmt>(stmt) && !As<ReturnStmt>(stmt) &&
        !As<IfStmt>(stmt) && !As<ForStmt>(stmt) && !As<WhileStmt>(stmt) && !As<YieldStmt>(stmt)) {
      admission.Require(false, BufferAdmissionProperty::StatementForm);
      return;
    }
    IRVisitor::VisitStmt(stmt);
  }

 protected:
  void VisitVarLike_(const VarPtr&) override {}

  void VisitBinaryExpr_(const BinaryExprPtr& expr) override {
    admission.Require(!GetScalarDtype(expr).IsUnsignedInt() && !GetScalarDtype(expr->left_).IsUnsignedInt() &&
                          !GetScalarDtype(expr->right_).IsUnsignedInt(),
                      BufferAdmissionProperty::ScalarArithmetic);
    IRVisitor::VisitBinaryExpr_(expr);
  }

  void VisitUnaryExpr_(const UnaryExprPtr& expr) override {
    const auto source = GetScalarDtype(expr->operand_);
    const auto target = GetScalarDtype(expr);
    if (As<Cast>(expr)) {
      admission.Require((!source.IsUnsignedInt() && !target.IsUnsignedInt()) || source == DataType::INDEX ||
                            source == target ||
                            (!source.IsFloat() && target != DataType::INDEX && !target.IsFloat() &&
                             source.GetBit() == target.GetBit()),
                        BufferAdmissionProperty::ScalarArithmetic);
    } else {
      admission.Require(!source.IsUnsignedInt() && !target.IsUnsignedInt(),
                        BufferAdmissionProperty::ScalarArithmetic);
    }
    IRVisitor::VisitUnaryExpr_(expr);
  }

  void VisitStmt_(const AssignStmtPtr& assign) override {
    if (As<TupleType>(assign->var_->GetType())) {
      const auto result = tuples_.Get(assign->var_);
      supported &= result && result->call == assign->value_;
      if (result) {
        for (const auto& output : result->outputs) CheckType(output->GetType());
        VisitExpr(result->call);
      }
      return;
    }
    if (tuples_.IsProjection(assign)) {
      VisitExpr(assign->var_);
      return;
    }
    if (const auto call = As<Call>(assign->value_)) {
      const auto source = As<TileType>(call->GetType());
      const auto result = As<TileType>(assign->var_->GetType());
      const auto source_memory = source ? source->memref_.value_or(nullptr) : nullptr;
      const auto result_memory = result ? result->memref_.value_or(nullptr) : nullptr;
      if (source_memory && source_memory->slot_index_ && source_memory->size_ == 0) {
        // InitMemRef resolves assignment destinations, while operation result
        // types may retain their frontend slot template. Lowering obtains the
        // physical descriptor from that destination. Accept only an identical
        // logical template bound to the same resolved family and selector.
        // The template may remain pinned: InitMemRef clears that declaration
        // marker only on the destination and preserves isolation on tile.alloc.
        if (!ConstantEquals(source_memory->byte_offset_, 0) || !result_memory ||
            !result_memory->slot_index_ || source_memory->base_ != result_memory->base_ ||
            source_memory->slot_count_ != result_memory->slot_count_ ||
            !structural_equal(*source_memory->slot_index_, *result_memory->slot_index_) ||
            !structural_equal(CloneTypeWithoutMemRef(source), CloneTypeWithoutMemRef(result))) {
          admission.Require(false, BufferAdmissionProperty::OperationRecipe);
          return;
        }
        assigned_slot_types_.emplace(call.get(), result);
      }
    }
    if (auto call = As<Call>(assign->value_); IsOp(call, "tile.matmul_acc") && call->args_.size() >= 3) {
      const auto output = As<TileType>(assign->var_->GetType());
      const auto memory = output ? output->memref_.value_or(nullptr) : nullptr;
      if (memory && (memory->slot_count_ > 1 || memory->slot_index_)) {
        const auto lhs = As<TileType>(call->args_[1]->GetType());
        const auto rhs = As<TileType>(call->args_[2]->GetType());
        if (!lhs || !rhs) {
          admission.Require(false, BufferAdmissionProperty::OperationRecipe);
          return;
        }
        const auto output_valid = ValidShape(output);
        const auto lhs_valid = ValidShape(lhs);
        const auto rhs_valid = ValidShape(rhs);
        // A narrowed product needs a write view; the initial slot adapter keeps
        // one complete descriptor rather than manufacturing partial slot aliases.
        admission.Require(output_valid.size() == 2 && lhs_valid.size() == 2 && rhs_valid.size() == 2 &&
                              structural_equal(output_valid[0], lhs_valid[0]) &&
                              structural_equal(output_valid[1], rhs_valid[1]),
                          BufferAdmissionProperty::OperationRecipe);
      }
    }
    if (auto call = As<Call>(assign->value_); IsOp(call, "tile.alloc") && call->args_.size() == 2) {
      if (auto capacity = As<ConstInt>(call->args_[1]); capacity && capacity->value_ > 0) {
        allocation_capacities_.emplace(assign->var_.get(), static_cast<uint64_t>(capacity->value_));
      }
    }
    IRVisitor::VisitStmt_(assign);
    if (!admission.supported()) return;
    if (As<ScalarType>(assign->var_->GetType())) {
      scalar_multiples_.emplace(assign->var_.get(), KnownAlignmentMultiple(assign->value_));
      step_proof_.Define(assign->var_, assign->value_);
    }
    if (IsMatrixType(assign->var_->GetType())) {
      ExprPtr source = assign->value_;
      if (auto call = As<Call>(source);
          call && (IsOp(call, "tile.matmul_acc") || IsOp(call, "tile.transpose_view"))) {
        source = call->args_[0];
      }
      const auto alias = StableAlias(source);
      matrix_aliases_.emplace(assign->var_.get(), alias ? alias : assign->var_);
      return;
    }
    if (!As<TensorType>(assign->var_->GetType())) return;
    ExprPtr source = assign->value_;
    if (auto call = As<Call>(source); call && IsOp(call, "tile.store") && call->args_.size() == 3) {
      source = call->args_[2];
    }
    auto var = AsVarLike(source);
    auto alias = tensor_aliases_.find(var.get());
    if (alias != tensor_aliases_.end()) tensor_aliases_.emplace(assign->var_.get(), alias->second);
  }

  void VisitStmt_(const EvalStmtPtr& eval) override {
    if (auto call = As<Call>(eval->expr_); call && IsOp(call, "tile.set_validshape")) {
      // Conversion needs an explicit SSA result carrying the new logical valid shape.
      admission.Require(false, BufferAdmissionProperty::OperationRecipe);
      return;
    }
    IRVisitor::VisitStmt_(eval);
  }

  void VisitStmt_(const ReturnStmtPtr& ret) override {
    // Buffer lowering erases store-result and plain GM aliases. Its codegen ABI
    // requires the resulting parameter type to match the declared result,
    // including TensorView metadata introduced by earlier frontend passes.
    if (region_ || ret->value_.size() != return_types_.size()) {
      admission.Require(false, BufferAdmissionProperty::OperationRecipe);
      return;
    }
    for (size_t i = 0; i < ret->value_.size(); ++i) {
      auto value = AsVarLike(ret->value_[i]);
      auto alias = tensor_aliases_.find(value.get());
      if (alias == tensor_aliases_.end() ||
          !structural_equal(CloneTypeWithoutMemRef(alias->second->GetType()),
                            CloneTypeWithoutMemRef(return_types_[i]))) {
        admission.Require(false, BufferAdmissionProperty::OperationRecipe);
        return;
      }
    }
    IRVisitor::VisitStmt_(ret);
  }

  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitExpr(branch->condition_);
    Region then_region(branch->return_vars_);
    CheckRegion(branch->then_body_, then_region);
    Region else_region(branch->return_vars_);
    if (branch->else_body_) CheckRegion(*branch->else_body_, else_region);
    for (size_t i = 0; admission.supported() && i < branch->return_vars_.size(); ++i) {
      const auto& result = branch->return_vars_[i];
      CheckRegionType(result->GetType());
      if (NeedsStableAlias(result->GetType())) {
        const auto& alias = then_region.aliases[i];
        if (!AliasesAgree(alias, else_region.aliases[i])) {
          admission.Require(false, BufferAdmissionProperty::OperationRecipe);
          return;
        }
        AliasMap(result->GetType()).emplace(result.get(), alias);
      }
    }
  }

  void VisitStmt_(const YieldStmtPtr& yield) override {
    if (!region_ || yield->value_.size() != region_->results.size()) {
      admission.Require(false, BufferAdmissionProperty::OperationRecipe);
      return;
    }
    region_->yielded = true;
    for (size_t i = 0; admission.supported() && i < yield->value_.size(); ++i) {
      CheckRegionType(region_->results[i]->GetType());
      VisitExpr(yield->value_[i]);
      CheckRegionValid(yield->value_[i]->GetType(), region_->results[i]->GetType());
      // A region result cannot export a branch-local window handle. Window
      // carries require scalar metadata and explicit reconstruction first.
      admission.Require(!views_.IsWindow(yield->value_[i]), BufferAdmissionProperty::OperationRecipe);
      if (NeedsStableAlias(region_->results[i]->GetType())) {
        region_->aliases[i] = StableAlias(yield->value_[i]);
        admission.Require(region_->aliases[i] != nullptr, BufferAdmissionProperty::OperationRecipe);
      }
    }
  }

  void VisitStmt_(const ForStmtPtr& loop) override {
    admission.Require(GetScalarDtype(loop->loop_var_) == DataType::INDEX,
                      BufferAdmissionProperty::OperationRecipe);
    for (const auto& bound : {loop->start_, loop->stop_, loop->step_}) {
      VisitExpr(bound);
      const auto dtype = GetScalarDtype(bound);
      admission.Require(dtype == DataType::INDEX || dtype.IsSignedInt(),
                        BufferAdmissionProperty::OperationRecipe);
    }
    // Buffer conversion preserves the original loop; it needs positivity, not
    // a numerical trip count. In particular get_block_num is positive by its
    // operation contract. Unknown signed scalar steps remain unqualified.
    admission.Require(step_proof_.IsPositive(loop->step_), BufferAdmissionProperty::LoopStep);
    loop_slots_.emplace_back();
    CheckLoop(loop);
    loop_slots_.pop_back();
  }

  void VisitStmt_(const WhileStmtPtr& loop) override {
    loop_slots_.emplace_back();
    CheckLoop(loop);
    loop_slots_.pop_back();
  }

  void VisitExpr_(const SubmitPtr&) override {
    admission.Require(false, BufferAdmissionProperty::OperationRecipe);
  }
  void VisitExpr_(const CallPtr& call) override {
    if (As<GlobalVar>(call->op_)) {
      admission.Require(false, BufferAdmissionProperty::OperationRecipe);
      return;
    }
    const bool view_operation = IsOp(call, "tile.slice") || IsOp(call, "tile.assemble") ||
                                IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view") ||
                                IsOp(call, "tile.transpose_view") || IsOp(call, "tile.set_validshape");
    if (view_operation) {
      for (const auto& arg : call->args_) {
        const auto tile = As<TileType>(arg->GetType());
        const auto memory = tile ? tile->memref_.value_or(nullptr) : nullptr;
        if (memory && (memory->slot_count_ > 1 || memory->slot_index_)) {
          admission.Require(false, BufferAdmissionProperty::OperationRecipe);
          return;
        }
      }
    }
    // Selection must reject metadata that the final lowering cannot carry.
    if (!StripAttr(call->attrs_, kPipelineMembershipAttr).empty()) {
      admission.Require(false, BufferAdmissionProperty::OperationRecipe);
      return;
    }
    if (IsOp(call, "tile.load")) {
      has_load = true;
      CheckTransferParameter(call, 0, true);
      CheckTransferTypes(call, true);
      admission.Require(GetIntKwarg(call->kwargs_, "cache", 0) == 0,
                        BufferAdmissionProperty::OperationRecipe);
    } else if (IsOp(call, "tile.store")) {
      has_store = true;
      // The Buffer store recipe accepts only the three-operand form. Keep
      // explicit partition shapes on Tile IR before any storage is rewritten.
      admission.Require(call->args_.size() == 3, BufferAdmissionProperty::OperationRecipe);
      CheckTransferParameter(call, 2, false);
      CheckTransferTypes(call, false);
      const bool ordinary_store = GetIntKwarg(call->kwargs_, "atomic", 0) == 0;
      admission.Require(ordinary_store, BufferAdmissionProperty::AtomicStore);
      // Preserve the original conjunction's short-circuiting of kwarg checks.
      admission.Require(ordinary_store && GetIntKwarg(call->kwargs_, "st_phase", 0) == 0 &&
                            !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                            !GetKwargOr<bool>(call->kwargs_, "pre_relu", false) &&
                            GetIntKwarg(call->kwargs_, "cache", 0) == 0,
                        BufferAdmissionProperty::TransferOptions);
    } else if (IsOp(call, "tile.set_validshape")) {
      admission.Require(call->args_.size() == 3 && call->kwargs_.empty() &&
                            As<TileType>(call->GetType())->GetMemorySpace() == MemorySpace::Vec,
                        BufferAdmissionProperty::OperationRecipe);
      if (admission.supported()) {
        const auto source = views_.Get(call->args_[0]);
        // A setter reconstructs the original slice. Relabelled windows have no
        // such coordinate recipe and need a separate typed-parent conversion.
        admission.Require(!source.window || (source.root && source.slice),
                          BufferAdmissionProperty::OperationRecipe);
      }
    } else if (IsOp(call, "tile.slice") || IsOp(call, "tile.assemble")) {
      CheckWindowCall(call);
    } else if (IsOp(call, "tile.reshape") || IsOp(call, "tile.reinterpret_view")) {
      const auto source = call->args_.empty() ? nullptr : As<TileType>(call->args_[0]->GetType());
      const auto destination = As<TileType>(call->GetType());
      const auto source_bytes = TileBytes(source);
      // Row reductions use a contiguous column descriptor, not a padded
      // row-major [R,1] box. The existing typed-root owner preserves its bytes
      // when the upstream layout resolver relabels it as a single dense row.
      const bool vector_relabel =
          source && destination && IsOp(call, "tile.reshape") && !views_.IsWindow(call->args_[0]) &&
          views_.HasDenseRowPitch(call->args_[0]) &&
          backend::IsContiguousVectorReshape(backend::StaticLogicalBufferDescriptor(source),
                                             backend::StaticLogicalBufferDescriptor(destination));
      admission.Require(source && destination && source->GetMemorySpace() == MemorySpace::Vec &&
                            destination->GetMemorySpace() == MemorySpace::Vec &&
                            (vector_relabel || (source_bytes && source_bytes == TileBytes(destination) &&
                                                views_.HasDenseRowPitch(call->args_[0]) &&
                                                (!views_.IsWindow(call->args_[0]) ||
                                                 views_.Get(call->args_[0]).offsets.has_value()))),
                        BufferAdmissionProperty::OperationRecipe);
      if (IsOp(call, "tile.reshape")) {
        admission.Require(call->args_.size() == 2 && call->kwargs_.empty() && source && destination &&
                              source->dtype_ == destination->dtype_,
                          BufferAdmissionProperty::OperationRecipe);
      } else {
        admission.Require((call->args_.size() == 1 || call->args_.size() == 2) && call->kwargs_.size() == 1 &&
                              call->kwargs_[0].first == "dtype",
                          BufferAdmissionProperty::OperationRecipe);
      }
    } else if (IsOp(call, "tile.move") || IsOp(call, "tile.extract") || IsOp(call, "tile.transpose_view") ||
               IsOp(call, "tile.matmul") || IsOp(call, "tile.matmul_acc")) {
      CheckMatrixCall(call);
    } else if (IsOp(call, "tile.create")) {
      admission.Require(As<TileType>(call->GetType()) != nullptr, BufferAdmissionProperty::OperationRecipe);
    } else if (IsOp(call, "tile.get_block_idx") || IsOp(call, "tile.get_block_num") ||
               IsOp(call, "tile.get_subblock_idx")) {
      admission.Require(call->args_.empty() && call->kwargs_.empty(),
                        BufferAdmissionProperty::OperationRecipe);
    } else if (const auto* indexed = backend::FindLogicalBufferIndexedRecipe(call->op_->name_)) {
      admission.Require(backend::SupportsLogicalBufferIndexed(*indexed, call) &&
                            (!indexed->scatter || indexed_.HasExclusiveZeroBase(call)),
                        BufferAdmissionProperty::OperationRecipe);
      for (const auto& arg : call->args_)
        admission.Require(!views_.IsWindow(arg), BufferAdmissionProperty::OperationRecipe);
    } else if (const auto* conversion = backend::FindLogicalBufferConversionRecipe(call->op_->name_)) {
      admission.Require(backend::SupportsLogicalBufferConversion(*conversion, call),
                        BufferAdmissionProperty::OperationRecipe);
      for (const auto& arg : call->args_)
        admission.Require(!views_.IsWindow(arg), BufferAdmissionProperty::OperationRecipe);
    } else if (IsOp(call, "tile.transpose")) {
      admission.Require(backend::SupportsLogicalBufferTranspose(call),
                        BufferAdmissionProperty::OperationRecipe);
      // A logical window can retain a wider physical pitch than its descriptor.
      for (const auto& arg : call->args_)
        admission.Require(!views_.IsWindow(arg), BufferAdmissionProperty::OperationRecipe);
    } else if (const auto* axis_recipe = backend::FindLogicalBufferAxisRecipe(call->op_->name_)) {
      admission.Require(backend::SupportsLogicalBufferAxis(*axis_recipe, call),
                        BufferAdmissionProperty::OperationRecipe);
      // Native axis recipes below are dense allocations. A slice retains a
      // different physical pitch and needs its own reduction/broadcast recipe.
      for (const auto& arg : call->args_)
        admission.Require(!views_.IsWindow(arg), BufferAdmissionProperty::OperationRecipe);
    } else if (!IsOp(call, "tile.alloc")) {
      const auto* recipe = backend::FindLogicalBufferElementwiseRecipe(call->op_->name_);
      admission.Require(
          recipe && recipe->default_enabled && backend::SupportsLogicalBufferElementwise(*recipe, call),
          BufferAdmissionProperty::OperationRecipe);
      if (IsOp(call, "tile.full")) has_fill = true;
    }
    IRVisitor::VisitExpr_(call);
  }

 private:
  static constexpr int64_t kWindowAlignmentBytes = 32;

  // Cache scalar SSA facts and expression nodes once. Unknown values have no
  // guaranteed factor beyond one; arithmetic preserves only proven multiples.
  int64_t KnownAlignmentMultiple(const ExprPtr& expr) {
    const auto cached = expression_multiples_.find(expr.get());
    if (cached != expression_multiples_.end()) return cached->second;
    int64_t multiple = 1;
    if (const auto constant = As<ConstInt>(expr)) {
      multiple = std::gcd(constant->value_ % kWindowAlignmentBytes, kWindowAlignmentBytes);
    } else if (const auto var = AsVarLike(expr)) {
      const auto found = scalar_multiples_.find(var.get());
      if (found != scalar_multiples_.end()) multiple = found->second;
    } else if (const auto cast = As<Cast>(expr)) {
      const auto source = GetScalarDtype(cast->operand_);
      const auto target = GetScalarDtype(cast);
      const auto integer = [](DataType dtype) {
        return dtype == DataType::INDEX || dtype.IsSignedInt() || dtype.IsUnsignedInt();
      };
      if (integer(source) && integer(target) && source.GetBit() >= 5 && target.GetBit() >= 5) {
        multiple = KnownAlignmentMultiple(cast->operand_);
      }
    } else if (const auto binary = As<BinaryExpr>(expr);
               binary && (As<Add>(expr) || As<Sub>(expr) || As<Mul>(expr))) {
      const auto lhs = KnownAlignmentMultiple(binary->left_);
      const auto rhs = KnownAlignmentMultiple(binary->right_);
      multiple = As<Mul>(expr) ? std::gcd(lhs * rhs, kWindowAlignmentBytes) : std::gcd(lhs, rhs);
    }
    expression_multiples_.emplace(expr.get(), multiple);
    return multiple;
  }

  std::optional<int64_t> WindowOffsetResidue(const ExprPtr& offset, int64_t stride_bytes) {
    const auto stride = stride_bytes % kWindowAlignmentBytes;
    if (const auto constant = As<ConstInt>(offset)) {
      const auto residue =
          (constant->value_ % kWindowAlignmentBytes + kWindowAlignmentBytes) % kWindowAlignmentBytes;
      return residue * stride % kWindowAlignmentBytes;
    }
    if (KnownAlignmentMultiple(offset) * stride % kWindowAlignmentBytes == 0) return 0;
    return std::nullopt;
  }

  void CheckWindowCall(const CallPtr& call) {
    const bool slice = IsOp(call, "tile.slice");
    if (call->args_.size() < 3) {
      admission.Require(false, BufferAdmissionProperty::WindowGeometry);
      return;
    }
    const auto parent = As<TileType>(call->args_[0]->GetType());
    const auto window = As<TileType>(slice ? call->GetType() : call->args_[1]->GetType());
    const auto offsets = As<MakeTuple>(call->args_[2]);
    if (!parent || !window || parent->GetMemorySpace() != MemorySpace::Vec ||
        window->GetMemorySpace() != MemorySpace::Vec || parent->dtype_ != window->dtype_ ||
        !StaticShape(parent->shape_) || !StaticShape(window->shape_) || !offsets ||
        offsets->elements_.size() != 2) {
      admission.Require(false, BufferAdmissionProperty::WindowGeometry);
      return;
    }
    // Explicit-valid slices and assemble destinations bypass slice
    // canonicalization. Prove their byte address alignment independently,
    // using the inherited parent pitch rather than the window's own width.
    const auto row_bytes = views_.Get(call->args_[0]).row_bytes;
    const auto row_residue =
        row_bytes ? WindowOffsetResidue(offsets->elements_[0], *row_bytes) : std::nullopt;
    const auto col_residue =
        WindowOffsetResidue(offsets->elements_[1], static_cast<int64_t>(window->dtype_.GetBit() / 8));
    admission.Require(
        row_residue && col_residue && (*row_residue + *col_residue) % kWindowAlignmentBytes == 0,
        BufferAdmissionProperty::WindowGeometry);
    for (size_t i = 0; i < 2; ++i) {
      const auto offset = As<ConstInt>(offsets->elements_[i]);
      const auto scalar = As<ScalarType>(offsets->elements_[i]->GetType());
      const auto limit = As<ConstInt>(parent->shape_[i])->value_ - As<ConstInt>(window->shape_[i])->value_;
      // Dynamic bounds remain the slice/assemble caller's precondition. They
      // never establish a precise range for alias analysis.
      admission.Require(limit >= 0 && scalar &&
                            (scalar->dtype_ == DataType::INDEX || scalar->dtype_.IsSignedInt()) &&
                            (!offset || (offset->value_ >= 0 && offset->value_ <= limit)),
                        BufferAdmissionProperty::WindowGeometry);
    }
    if (slice) {
      admission.Require(call->kwargs_.empty() && call->args_.size() <= 5,
                        BufferAdmissionProperty::WindowGeometry);
      const auto parent_view = views_.Get(call->args_[0]);
      admission.Require(!parent_view.window || parent_view.root != nullptr,
                        BufferAdmissionProperty::WindowGeometry);
      if (call->args_.size() == 5) {
        const auto dropped = As<MakeTuple>(call->args_[4]);
        admission.Require(dropped && dropped->elements_.empty(), BufferAdmissionProperty::WindowGeometry);
      }
    } else {
      admission.Require(call->args_.size() == 3 && !views_.IsWindow(call->args_[0]) &&
                            !GetOptionalDoubleKwarg(call->kwargs_, "pre_quant") &&
                            !GetKwargOr<bool>(call->kwargs_, "pre_relu", false) &&
                            GetIntKwarg(call->kwargs_, "cache", 0) == 0,
                        BufferAdmissionProperty::WindowGeometry);
      // Native Vec TMOV preserves row stride; copying a strided window into
      // a destination with a different pitch needs a separate transfer recipe.
      const auto source_view = views_.Get(call->args_[1]);
      admission.Require(source_view.row_bytes &&
                            source_view.row_bytes == views_.Get(call->args_[0]).row_bytes &&
                            (!source_view.window || (source_view.root && source_view.slice)),
                        BufferAdmissionProperty::WindowGeometry);
    }
    if (!views_.IsWindow(call->args_[0])) {
      const auto memory = parent->memref_.value_or(nullptr);
      if (memory) {
        const auto [entry, inserted] = window_parents_.emplace(memory->base_.get(), parent);
        // One symbolic allocation can have only one native typed parent.
        // Conflicting parent interpretations need additional legalization.
        admission.Require(inserted || (entry->second->dtype_ == parent->dtype_ &&
                                       SameShape(entry->second->shape_, parent->shape_)),
                          BufferAdmissionProperty::WindowGeometry);
      }
    }
  }

  bool QualifySlot(const TileTypePtr& tile, const MemRefPtr& memory) {
    const auto space = tile->GetMemorySpace();
    if (!space || (*space != MemorySpace::Vec && !backend::IsMatrixBufferSpace(*space)) || !memory->base_ ||
        !memory->slot_index_ || !*memory->slot_index_ || memory->slot_count_ == 0 || memory->slot_count_ > 16)
      return false;
    const auto capacity = allocation_capacities_.find(memory->base_.get());
    const auto bytes = TileBytes(tile);
    const auto origin = FixedMultiBufferBase(memory);
    const auto* context = PassContext::Current();
    const auto* target = context ? context->GetBackend() : backend::GetBackend();
    const auto alignment = target->GetMemAlignment(*space);
    if (!bytes || !origin || *origin != 0 || !alignment || *bytes % alignment != 0 ||
        memory->size_ != *bytes || capacity == allocation_capacities_.end() ||
        capacity->second / memory->slot_count_ != *bytes || capacity->second % memory->slot_count_ != 0) {
      return false;
    }
    const auto valid = ValidShape(tile);
    for (const auto& value : valid) {
      if (!As<ConstInt>(value) || As<ConstInt>(value)->value_ <= 0) return false;
    }
    // Earlier shape qualification proves whole dense/fractal boxes. Their
    // physical bytes equal this dense byte count. Keep every layout/valid field
    // identical across the family; later lowering rechecks final placement.
    auto descriptor = CloneTypeWithoutMemRef(tile);
    auto [entry, inserted] = slot_descriptors_.emplace(memory->base_.get(), descriptor);
    return inserted || structural_equal(entry->second, descriptor);
  }

  static std::optional<uint64_t> TileBytes(const TileTypePtr& tile) {
    if (!tile || !StaticShape(tile->shape_)) return std::nullopt;
    const std::vector<int64_t> shape{As<ConstInt>(tile->shape_[0])->value_,
                                     As<ConstInt>(tile->shape_[1])->value_};
    return backend::PhysicalBufferBytes(std::make_shared<BufferType>(shape, tile->dtype_, MemorySpace::Vec));
  }

  static std::vector<ExprPtr> ValidShape(const TileTypePtr& tile) {
    const auto view = tile_view_semantics::GetEffectiveTileView(*tile);
    return view.valid_shape.empty() ? tile->shape_ : view.valid_shape;
  }

  void CheckTransferTypes(const CallPtr& call, bool load) {
    if (call->args_.size() < 3) {
      admission.Require(false, BufferAdmissionProperty::TransferTypes);
      return;
    }
    const auto tile = As<TileType>(load ? call->GetType() : call->args_[0]->GetType());
    const auto tensor = As<TensorType>(call->args_[load ? 0 : 2]->GetType());
    if (!tile || !tensor) {
      admission.Require(false, BufferAdmissionProperty::TransferTypes);
      return;
    }
    const auto space = tile->GetMemorySpace();
    // Reduction columns are on-chip operands only. The ordinary GM recipe
    // cannot transfer a contiguous column into a strided row-major tensor.
    admission.Require(
        !backend::IsColumnVectorBufferTile(tile, tile_view_semantics::GetEffectiveTileView(*tile)),
        BufferAdmissionProperty::TransferTypes);
    if (space == MemorySpace::Vec || (load && space == MemorySpace::Mat)) {
      admission.Require(tile->dtype_ == tensor->dtype_, BufferAdmissionProperty::TransferTypes);
    } else {
      admission.Require(
          !load && space == MemorySpace::Acc && CubeWritebackSupportsDataType(tile->dtype_, tensor->dtype_),
          BufferAdmissionProperty::TransferTypes);
    }
  }

  void CheckMatrixCall(const CallPtr& call) {
    const auto destination = As<TileType>(call->GetType());
    const auto source = call->args_.empty() ? nullptr : As<TileType>(call->args_[0]->GetType());
    if (!source || !destination) {
      admission.Require(false, BufferAdmissionProperty::MatrixRecipe);
      return;
    }
    if (IsOp(call, "tile.move") || IsOp(call, "tile.extract")) {
      admission.Require(source->GetMemorySpace() == MemorySpace::Mat &&
                            (destination->GetMemorySpace() == MemorySpace::Left ||
                             destination->GetMemorySpace() == MemorySpace::Right) &&
                            source->dtype_ == destination->dtype_,
                        BufferAdmissionProperty::MatrixRecipe);
      if (IsOp(call, "tile.move")) {
        admission.Require(SameShape(source->shape_, destination->shape_) &&
                              SameShape(ValidShape(source), ValidShape(destination)),
                          BufferAdmissionProperty::MatrixRecipe);
      } else {
        admission.Require(call->args_.size() == 4, BufferAdmissionProperty::MatrixRecipe);
      }
    } else if (IsOp(call, "tile.transpose_view")) {
      admission.Require(
          call->args_.size() == 1 && call->kwargs_.empty() && source->GetMemorySpace() == MemorySpace::Mat &&
              destination->GetMemorySpace() == MemorySpace::Mat && source->dtype_ == destination->dtype_ &&
              TileBytes(source) && TileBytes(source) == TileBytes(destination),
          BufferAdmissionProperty::MatrixRecipe);
    } else {
      const bool accumulate = IsOp(call, "tile.matmul_acc");
      const size_t lhs_index = accumulate ? 1 : 0;
      if (call->args_.size() < lhs_index + 2) {
        admission.Require(false, BufferAdmissionProperty::MatrixRecipe);
        return;
      }
      const auto lhs = As<TileType>(call->args_[lhs_index]->GetType());
      const auto rhs = As<TileType>(call->args_[lhs_index + 1]->GetType());
      admission.Require(lhs && rhs && call->kwargs_.empty() && call->args_.size() <= (accumulate ? 4 : 2) &&
                            destination->GetMemorySpace() == MemorySpace::Acc,
                        BufferAdmissionProperty::MatrixRecipe);
      if (!admission.supported()) return;
      admission.Require(lhs->GetMemorySpace() == MemorySpace::Left &&
                            rhs->GetMemorySpace() == MemorySpace::Right && lhs->dtype_ == rhs->dtype_ &&
                            backend::IsCubeOperandDtype(lhs->dtype_) &&
                            destination->dtype_ == MatmulAccumulatorDataType(lhs->dtype_, rhs->dtype_),
                        BufferAdmissionProperty::MatrixRecipe);
      if (accumulate)
        admission.Require(source->GetMemorySpace() == MemorySpace::Acc,
                          BufferAdmissionProperty::MatrixRecipe);
    }
  }

  void CheckRegionType(const TypePtr& type) {
    admission.Require(As<TileType>(type) || As<TensorType>(type) || As<ScalarType>(type),
                      BufferAdmissionProperty::RegionType);
    CheckType(type);
    if (auto tile = As<TileType>(type)) {
      const auto memory = tile->memref_.value_or(nullptr);
      admission.Require(!memory || (memory->slot_count_ == 1 && !memory->slot_index_),
                        BufferAdmissionProperty::RegionType);
    }
  }

  void CheckRegionValid(const TypePtr& source, const TypePtr& destination) {
    const auto from = As<TileType>(source);
    const auto to = As<TileType>(destination);
    if (!from || !to || !(NeedsRuntimeBufferValid(from) || NeedsRuntimeBufferValid(to))) return;
    // Canonical storage carries bytes, not a choice of valid expressions.
    // Changing valid state must be carried explicitly as scalar SSA and used
    // to reconstruct a view; silently inheriting either arm's metadata is wrong.
    admission.Require(SameShape(ValidShape(from), ValidShape(to)), BufferAdmissionProperty::RegionValidState);
  }

  struct Region {
    explicit Region(const std::vector<VarPtr>& results) : results(results), aliases(results.size()) {}
    const std::vector<VarPtr>& results;
    std::vector<VarPtr> aliases;
    bool yielded = false;
  };

  static bool IsMatrixType(const TypePtr& type) {
    const auto tile = As<TileType>(type);
    return tile && tile->GetMemorySpace() && backend::IsMatrixBufferSpace(*tile->GetMemorySpace());
  }

  static bool NeedsStableAlias(const TypePtr& type) { return As<TensorType>(type) || IsMatrixType(type); }

  std::unordered_map<const Var*, VarPtr>& AliasMap(const TypePtr& type) {
    return IsMatrixType(type) ? matrix_aliases_ : tensor_aliases_;
  }

  // Selection precedes storage legalization. Reuse its liveness-checked Acc
  // phi plan instead of demanding that a fresh seed already have the carry's
  // SSA identity. Planning is read-only and lazy, at most once per function;
  // the existing semantic pass remains responsible for applying the rewrites.
  bool AliasesAgree(const VarPtr& left, const VarPtr& right) {
    if (!left || !right) return false;
    if (left == right) return true;
    const auto a = As<TileType>(left->GetType());
    const auto b = As<TileType>(right->GetType());
    if (!a || !b || a->GetMemorySpace() != MemorySpace::Acc || b->GetMemorySpace() != MemorySpace::Acc)
      return false;
    if (!accumulator_plan_) accumulator_plan_ = memory_reuse::PlanAccumulatorIfPhis(body_);
    auto planned = [&](const VarPtr& value, const TileTypePtr& original) {
      const auto found = accumulator_plan_->find(value);
      const auto type = found == accumulator_plan_->end() ? original : As<TileType>(found->second);
      return type ? type->memref_.value_or(nullptr) : nullptr;
    };
    return memory_reuse::SamePhysicalWindow(planned(left, a), planned(right, b));
  }

  StmtPtr body_;
  std::optional<std::map<VarPtr, TypePtr>> accumulator_plan_;

  VarPtr StableAlias(const ExprPtr& value) {
    const auto var = AsVarLike(value);
    auto& aliases = AliasMap(value->GetType());
    const auto alias = aliases.find(var.get());
    return alias == aliases.end() ? nullptr : alias->second;
  }

  void CheckRegion(const StmtPtr& body, Region& region) {
    auto* outer = region_;
    region_ = &region;
    VisitStmt(body);
    admission.Require(region.results.empty() || region.yielded, BufferAdmissionProperty::RegionYield);
    region_ = outer;
  }

  template <typename LoopPtr>
  void CheckLoop(const LoopPtr& loop) {
    if (!admission.supported() || loop->iter_args_.size() != loop->return_vars_.size()) {
      admission.Require(false, BufferAdmissionProperty::LoopCarry);
      return;
    }
    // SSA identities are unique across regions. Retaining each normalized alias
    // once avoids copying an ever-growing map at nested branches and loops.
    for (size_t i = 0; admission.supported() && i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      CheckRegionType(argument->GetType());
      CheckRegionType(loop->return_vars_[i]->GetType());
      VisitExpr(argument->initValue_);
      CheckRegionValid(argument->initValue_->GetType(), argument->GetType());
      CheckRegionValid(argument->GetType(), loop->return_vars_[i]->GetType());
      admission.Require(!views_.IsWindow(argument->initValue_), BufferAdmissionProperty::LoopCarry);
      if (NeedsStableAlias(argument->GetType())) {
        const auto alias = StableAlias(argument->initValue_);
        if (!alias) {
          admission.Require(false, BufferAdmissionProperty::LoopCarry);
          return;
        }
        AliasMap(argument->GetType()).emplace(argument.get(), alias);
      }
    }
    if constexpr (std::is_same_v<LoopPtr, WhileStmtPtr>) VisitExpr(loop->condition_);
    Region region(loop->return_vars_);
    CheckRegion(loop->body_, region);
    for (size_t i = 0; admission.supported() && i < loop->iter_args_.size(); ++i) {
      const auto& argument = loop->iter_args_[i];
      if (NeedsStableAlias(argument->GetType())) {
        const auto initial = StableAlias(argument->initValue_);
        if (!AliasesAgree(region.aliases[i], initial)) {
          admission.Require(false, BufferAdmissionProperty::LoopCarry);
          return;
        }
        AliasMap(argument->GetType()).emplace(loop->return_vars_[i].get(), initial);
      }
    }
  }

  void CheckTransferParameter(const CallPtr& call, size_t index, bool load) {
    if (index >= call->args_.size()) {
      admission.Require(false, BufferAdmissionProperty::TransferParameter);
      return;
    }
    const auto value = AsVarLike(call->args_[index]);
    const auto alias = tensor_aliases_.find(value.get());
    if (alias == tensor_aliases_.end()) {
      admission.Require(false, BufferAdmissionProperty::TransferParameter);
      return;
    }
    const auto direction = directions_.at(alias->second.get());
    admission.Require(load ? direction != ParamDirection::Out : direction != ParamDirection::In,
                      BufferAdmissionProperty::TransferParameter);
  }

  std::unordered_map<const Var*, uint64_t> allocation_capacities_;
  std::unordered_map<const Var*, TypePtr> slot_descriptors_;
  std::vector<std::unordered_map<const Var*, ExprPtr>> loop_slots_;
  Region* region_ = nullptr;
  const std::vector<TypePtr>& return_types_;
  BufferViewAnalysis views_;
  BufferIndexedAnalysis indexed_;
  BufferTupleResults tuples_;
  std::unordered_map<const Var*, int64_t> scalar_multiples_;
  std::unordered_map<const Expr*, int64_t> expression_multiples_;
  std::unordered_map<const Var*, TileTypePtr> window_parents_;
  std::unordered_map<const Var*, ParamDirection> directions_;
  std::unordered_map<const Var*, VarPtr> tensor_aliases_;
  std::unordered_map<const Var*, VarPtr> matrix_aliases_;
  std::unordered_set<const Type*> checked_types_;
  std::unordered_map<const Expr*, TypePtr> assigned_slot_types_;
  PositiveLoopStepProof step_proof_;
};

FunctionPtr SelectBufferFunction(const FunctionPtr& function, bool device_helper) {
  if (!IsInCoreType(function->func_type_) || function->ir_stage_ == FunctionIRStage::Buffer ||
      function->HasAttr(kAttrBufferIrSelected)) {
    return function;
  }
  BufferKernelSupport support(function);
  auto function_context = support.admission.At(function.get());
  const auto* context = PassContext::Current();
  support.admission.Require(!device_helper, BufferAdmissionProperty::DeviceHelper);
  support.admission.Require(!context || context->GetEnableBufferIR(), BufferAdmissionProperty::Disabled);
  support.admission.Require(backend::BackendConfig::IsConfigured(), BufferAdmissionProperty::Backend);
  if (support.admission.supported()) {
    const auto arch = backend::BackendConfig::GetBackend()->GetHandler()->GetPtoTargetArch();
    support.admission.Require(arch == "a2a3", BufferAdmissionProperty::Target);
  }
  for (const auto& param : function->params_) {
    auto parameter_context = support.admission.At(param.get());
    // Device Tile parameters still require a separate helper ABI.
    support.admission.Require(As<TensorType>(param->GetType()) || As<ScalarType>(param->GetType()),
                              BufferAdmissionProperty::ParameterABI);
    support.CheckType(param->GetType());
  }
  for (const auto& type : function->return_types_) {
    support.admission.Require(As<TensorType>(type) != nullptr, BufferAdmissionProperty::ReturnABI);
    support.CheckType(type);
  }
  support.VisitStmt(function->body_);
  if (support.admission.supported()) {
    // Selecting Buffer must remain executable even when optional autosync later
    // declines. Share the existing downstream canonical-slot qualification.
    CanonicalSlotCollector slots;
    slots.VisitStmt(function->body_);
    auto slot_context = support.admission.At(
        slots.first_unsupported ? static_cast<const IRNode*>(slots.first_unsupported.get()) : function.get());
    support.admission.Require(slots.unsupported_bases.empty(), BufferAdmissionProperty::CanonicalSlots);
  }
  if (support.admission.supported()) {
    const BufferValidAnalysis valid(function->body_);
    auto valid_context = support.admission.At(valid.first_obligation()
                                                  ? static_cast<const IRNode*>(valid.first_obligation().get())
                                                  : function.get());
    support.admission.Require(valid.supported(), BufferAdmissionProperty::DynamicValidRecipe);
  }
  support.admission.Require((support.has_load || support.has_fill) && support.has_store,
                            BufferAdmissionProperty::KernelShape);
  auto selected = std::make_shared<Function>(*function);
  selected->attrs_.emplace_back(kAttrBufferIrSelected, support.admission.supported());
  support.admission.AppendTo(*selected);
  return selected;
}

// A GM-shaped function called from device code is still a helper, not a
// separately dispatched kernel. Keep both caller and callee out of this rollout.
class DeviceHelperCollector : public IRVisitor {
 public:
  std::unordered_set<std::string> callees;

 protected:
  void VisitVarLike_(const VarPtr&) override {}
  void VisitExpr_(const CallPtr& call) override { Collect(call); }
  void VisitExpr_(const SubmitPtr& submit) override {
    Collect(submit);
    for (const auto& dep : submit->deps_) VisitExpr(dep);
  }

 private:
  template <typename CallLike>
  void Collect(const CallLike& call) {
    if (auto global = As<GlobalVar>(call->op_)) callees.insert(global->name_);
    for (const auto& arg : call->args_) VisitExpr(arg);
  }
};

ProgramPtr SelectBufferFunctions(const ProgramPtr& program) {
  DeviceHelperCollector helpers;
  for (const auto& [global, function] : program->functions_) {
    if (IsInCoreType(function->func_type_)) helpers.VisitStmt(function->body_);
  }
  std::vector<FunctionPtr> functions;
  functions.reserve(program->functions_.size());
  for (const auto& [global, function] : program->functions_) {
    functions.push_back(SelectBufferFunction(function, helpers.callees.count(function->name_) != 0));
  }
  return std::make_shared<Program>(functions, program->name_, program->span_);
}

}  // namespace

namespace pass {

Pass SelectBufferIRFunctions() {
  return CreateProgramPass(SelectBufferFunctions, "SelectBufferIRFunctions",
                           kSelectBufferIRFunctionsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
