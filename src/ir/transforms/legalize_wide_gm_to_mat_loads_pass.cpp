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

/**
 * @file legalize_wide_gm_to_mat_loads_pass.cpp
 * @brief Rebase GM rows whose leading dimension cannot be encoded by GM->Mat tload.
 *
 * The A2/A3 fractal load path silently truncates an ND source row stride at
 * 2^16 elements. Once TensorView strides, tile placement and MemRefs are all
 * known, replace only affected GM->Mat tile.load calls with an allocated
 * destination plus one pointer-rebased row load per logical row. Each compact
 * source view has stride [C, 1], so the parent leading dimension is used only
 * in index arithmetic and never reaches the backend field.
 */

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/auto_name_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
#include "pypto/ir/transforms/utils/tensor_view_semantics.h"
#include "pypto/ir/type.h"
#include "pypto/ir/type_inference.h"

namespace pypto {
namespace ir {
namespace {

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

MakeTuplePtr Tuple(std::vector<ExprPtr> values, const Span& span) {
  return std::make_shared<MakeTuple>(std::move(values), span);
}

std::string TempName(const VarPtr& result, const std::string& role, size_t index) {
  return auto_name::BuildName(auto_name::GetBaseName(result->name_hint_), role, "wide_mat_load",
                              static_cast<int>(index));
}

class LegalizeWideGmToMatLoadsMutator : public IRMutator {
 public:
  explicit LegalizeWideGmToMatLoadsMutator(const backend::BackendHandler& handler) : handler_(handler) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto load = As<Call>(op->value_);
    if (!load || !IsOp(load, "tile.load") || load->args_.size() != 4) {
      return IRMutator::VisitStmt_(op);
    }

    auto dst = As<TileType>(op->var_->GetType());
    auto src = AsTensorTypeLike(load->args_[0]->GetType());
    INTERNAL_CHECK_SPAN(dst && src, op->span_)
        << "tile.load must have a TensorType source and TileType result";
    if (dst->memory_space_ != MemorySpace::Mat) return IRMutator::VisitStmt_(op);

    const auto max_stride = handler_.GetMaxGmToMatRowStrideElements(src->dtype_);
    if (!max_stride.has_value()) return IRMutator::VisitStmt_(op);
    INTERNAL_CHECK_SPAN(*max_stride > 0, op->span_)
        << "GetMaxGmToMatRowStrideElements must return a positive limit";
    if (src->shape_.size() != 2 || dst->shape_.size() != 2) {
      return IRMutator::VisitStmt_(op);
    }

    TensorLayout layout = TensorLayout::ND;
    std::vector<ExprPtr> strides;
    if (src->tensor_view_.has_value()) {
      layout = src->tensor_view_->layout;
      strides = src->tensor_view_->stride;
    }
    if (layout != TensorLayout::ND) return IRMutator::VisitStmt_(op);
    if (strides.empty()) {
      strides = tensor_view_semantics::BuildLogicalStridesFromLayout(src->shape_, layout);
    }
    INTERNAL_CHECK_SPAN(strides.size() == 2, op->span_);
    auto row_stride = As<ConstInt>(strides[0]);
    if (!row_stride || row_stride->value_ <= static_cast<int64_t>(*max_stride)) {
      return IRMutator::VisitStmt_(op);
    }

    auto offsets = As<MakeTuple>(load->args_[1]);
    auto shapes = As<MakeTuple>(load->args_[2]);
    auto valid = As<MakeTuple>(load->args_[3]);
    INTERNAL_CHECK_SPAN(offsets && shapes && valid && offsets->elements_.size() == 2 &&
                            shapes->elements_.size() == 2 && valid->elements_.size() == 2,
                        op->span_)
        << "LegalizeWideGmToMatLoads requires literal rank-2 tile.load windows";
    auto physical_rows = As<ConstInt>(shapes->elements_[0]);
    auto physical_cols = As<ConstInt>(shapes->elements_[1]);
    INTERNAL_CHECK_SPAN(physical_rows && physical_cols, op->span_)
        << "LegalizeWideGmToMatLoads requires a static physical tile shape";
    INTERNAL_CHECK_SPAN(dst->memref_.has_value(), op->span_)
        << "LegalizeWideGmToMatLoads requires an allocated destination after InitMemRef";

    std::vector<StmtPtr> outer;
    outer.push_back(MakeDestinationAllocation(op, dst));

    auto row = std::make_shared<Var>(TempName(op->var_, "row", temp_counter_++),
                                     std::make_shared<ScalarType>(DataType::INDEX), op->span_);
    auto src_row = MakeAdd(offsets->elements_[0], row, op->span_);
    auto src_offset = Tuple({src_row, offsets->elements_[1]}, op->span_);
    auto dst_offset = Tuple({row, Index(0, op->span_)}, op->span_);
    auto row_shape = Tuple({Index(1, op->span_), Index(physical_cols->value_, op->span_)}, op->span_);
    auto row_valid = Tuple({Index(1, op->span_), valid->elements_[1]}, op->span_);

    std::vector<std::pair<std::string, std::any>> kwargs;
    kwargs.emplace_back("cache", load->GetKwarg<int>("cache", 0));
    auto fragment = OpRegistry::GetInstance().CreateInternal(
        "tile.load_rebased_row", {op->var_, load->args_[0], dst_offset, src_offset, row_shape, row_valid},
        kwargs, op->span_);
    auto fragment_var =
        std::make_shared<Var>(TempName(op->var_, "loaded", temp_counter_++), fragment->GetType(), op->span_);
    auto fragment_stmt = std::make_shared<AssignStmt>(fragment_var, fragment, op->span_);
    outer.push_back(std::make_shared<ForStmt>(row, Index(0, op->span_), valid->elements_[0],
                                              Index(1, op->span_), std::vector<IterArgPtr>{}, fragment_stmt,
                                              std::vector<VarPtr>{}, op->span_));
    return SeqStmts::Flatten(std::move(outer), op->span_);
  }

 private:
  StmtPtr MakeDestinationAllocation(const AssignStmtPtr& original, const TileTypePtr& dst) const {
    auto shape = Tuple(dst->shape_, original->span_);
    std::vector<std::pair<std::string, std::any>> kwargs = {{"dtype", dst->dtype_},
                                                            {"target_memory", *dst->memory_space_}};
    auto inferred = OpRegistry::GetInstance().Create("tile.create", {shape}, kwargs, original->span_);
    auto create = std::make_shared<Call>(inferred->op_, inferred->args_, inferred->kwargs_, inferred->attrs_,
                                         original->var_->GetType(), original->span_);
    return std::make_shared<AssignStmt>(original->var_, create, original->span_, original->leading_comments_);
  }

  const backend::BackendHandler& handler_;
  size_t temp_counter_ = 0;
};

FunctionPtr TransformLegalizeWideGmToMatLoads(const FunctionPtr& func) {
  if (!func || (func->level_.has_value() && *func->level_ == Level::HOST) ||
      !backend::BackendConfig::IsConfigured()) {
    return func;
  }
  const auto* context = PassContext::Current();
  const auto* handler =
      context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (!handler) return func;
  LegalizeWideGmToMatLoadsMutator mutator(*handler);
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LegalizeWideGmToMatLoads() {
  return CreateFunctionPass(TransformLegalizeWideGmToMatLoads, "LegalizeWideGmToMatLoads",
                            kLegalizeWideGmToMatLoadsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
