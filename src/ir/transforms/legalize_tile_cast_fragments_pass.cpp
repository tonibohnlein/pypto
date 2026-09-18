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
 * @file legalize_tile_cast_fragments_pass.cpp
 * @brief Legalize target-restricted native casts after physical tile layout is known.
 *
 * A target may support a dtype pair but require a bounded physical column width
 * for each instruction. This pass keeps that capability in BackendHandler and
 * materializes it in Tile IR, after MemRefs and layouts are final: the original
 * result allocation becomes a tile.create, and each one-row source/destination
 * window is represented by pitch-preserving tile.slice views followed by one
 * internal destination-passing tile.cast_fragment. PTO codegen therefore has a
 * mechanical 1:1 lowering and contains no tail or target-shape decisions.
 */

#include <algorithm>
#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/cast_saturation.h"
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
#include "pypto/ir/transforms/utils/memref_utils.h"
#include "pypto/ir/transforms/utils/op_predicates.h"
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
  return auto_name::BuildName(auto_name::GetBaseName(result->name_hint_), role, "fragment",
                              static_cast<int>(index));
}

class LegalizeTileCastFragmentsMutator : public IRMutator {
 public:
  explicit LegalizeTileCastFragmentsMutator(const backend::BackendHandler& handler) : handler_(handler) {}

  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    auto cast = As<Call>(op->value_);
    if (!cast || !IsOp(cast, "tile.cast") || cast->args_.empty()) {
      return IRMutator::VisitStmt_(op);
    }

    auto src = As<TileType>(cast->args_[0]->GetType());
    auto dst = As<TileType>(op->var_->GetType());
    INTERNAL_CHECK_SPAN(src && dst, op->span_) << "tile.cast must have TileType source and result";
    const auto fragment_width = handler_.GetTcvtSafeFragmentWidth(src->dtype_, dst->dtype_);
    if (!fragment_width.has_value()) return IRMutator::VisitStmt_(op);
    INTERNAL_CHECK_SPAN(*fragment_width > 0, op->span_)
        << "GetTcvtSafeFragmentWidth must return a positive width";
    INTERNAL_CHECK_SPAN(src->shape_.size() == 2 && dst->shape_.size() == 2, op->span_)
        << "LegalizeTileCastFragments requires rank-2 Tile IR";
    auto physical_cols = As<ConstInt>(src->shape_[1]);
    INTERNAL_CHECK_SPAN(physical_cols, op->span_)
        << "LegalizeTileCastFragments requires a static physical column extent";
    const auto src_valid = GetValidShape(src);
    const auto dst_valid = GetValidShape(dst);
    INTERNAL_CHECK_SPAN(src_valid.size() == 2 && dst_valid.size() == 2, op->span_);
    auto static_valid_cols = As<ConstInt>(src_valid[1]);

    // A complete one-fragment frame and a complete integral number of hardware
    // fragments are already legal. Padded or runtime-valid frames still need
    // explicit fragments even when their physical width is aligned.
    if (static_valid_cols && static_valid_cols->value_ == physical_cols->value_ &&
        (physical_cols->value_ <= static_cast<int64_t>(*fragment_width) ||
         physical_cols->value_ % static_cast<int64_t>(*fragment_width) == 0)) {
      return IRMutator::VisitStmt_(op);
    }

    INTERNAL_CHECK_SPAN(src->memref_.has_value() && dst->memref_.has_value(), op->span_)
        << "LegalizeTileCastFragments must run after InitMemRef";
    INTERNAL_CHECK_SPAN(src->memory_space_ == MemorySpace::Vec && dst->memory_space_ == MemorySpace::Vec,
                        op->span_)
        << "LegalizeTileCastFragments supports Vec-resident native casts only";

    std::vector<StmtPtr> outer;
    outer.push_back(MakeDestinationAllocation(op, dst));

    struct Fragment {
      int64_t col;
      int64_t width;
      ExprPtr valid;
      ExprPtr nonempty;
    };
    std::vector<Fragment> fragments;
    for (int64_t col = 0; col < physical_cols->value_; col += *fragment_width) {
      const int64_t width = std::min<int64_t>(*fragment_width, physical_cols->value_ - col);
      if (static_valid_cols) {
        const int64_t valid = std::clamp<int64_t>(static_valid_cols->value_ - col, 0, width);
        if (valid == 0) continue;
        fragments.push_back(Fragment{col, width, Index(valid, op->span_), nullptr});
        continue;
      }

      // Compute runtime tail widths once, outside the row loop. max(valid-col,
      // 0) followed by min(..., width) is valid for every boundary, including
      // zero, an exact 128-wide fragment, and a residual one-element tail.
      auto remaining_expr =
          MakeMax(MakeSub(src_valid[1], Index(col, op->span_), op->span_), Index(0, op->span_), op->span_);
      auto remaining = BindScalar(outer, op->var_, "remaining", remaining_expr, op->span_);
      auto valid_expr = MakeMin(remaining, Index(width, op->span_), op->span_);
      auto valid = BindScalar(outer, op->var_, "valid", valid_expr, op->span_);
      fragments.push_back(Fragment{col, width, valid, MakeGt(valid, Index(0, op->span_), op->span_)});
    }

    auto row = std::make_shared<Var>(TempName(op->var_, "row", temp_counter_++),
                                     std::make_shared<ScalarType>(DataType::INDEX), op->span_);
    std::vector<StmtPtr> row_body;
    for (const auto& fragment : fragments) {
      auto body = MakeFragment(op, cast, row, fragment.col, fragment.width, fragment.valid);
      if (fragment.nonempty) {
        body =
            std::make_shared<IfStmt>(fragment.nonempty, body, std::nullopt, std::vector<VarPtr>{}, op->span_);
      }
      row_body.push_back(std::move(body));
    }
    auto loop_body = SeqStmts::Flatten(std::move(row_body), op->span_);
    outer.push_back(std::make_shared<ForStmt>(row, Index(0, op->span_), src_valid[0], Index(1, op->span_),
                                              std::vector<IterArgPtr>{}, loop_body, std::vector<VarPtr>{},
                                              op->span_));
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

  VarPtr BindScalar(std::vector<StmtPtr>& statements, const VarPtr& result, const std::string& role,
                    const ExprPtr& value, const Span& span) {
    auto var = std::make_shared<Var>(TempName(result, role, temp_counter_++), value->GetType(), span);
    statements.push_back(std::make_shared<AssignStmt>(var, value, span));
    return var;
  }

  StmtPtr MakeFragment(const AssignStmtPtr& original, const CallPtr& cast, const VarPtr& row, int64_t col,
                       int64_t width, const ExprPtr& valid_cols) {
    auto shape = Tuple({Index(1, original->span_), Index(width, original->span_)}, original->span_);
    auto offset = Tuple({row, Index(col, original->span_)}, original->span_);
    auto valid = Tuple({Index(1, original->span_), valid_cols}, original->span_);
    auto& registry = OpRegistry::GetInstance();

    // Keep fragment views MemRef-less. tile.slice already aliases its parent
    // SSA and lowers directly to pto.subview; attaching the parent's MemRef
    // would make generic per-Var codegen predeclare a dense alias tile that the
    // view does not own.
    auto src_slice = registry.Create("tile.slice", {cast->args_[0], shape, offset, valid}, original->span_);
    auto src_var = std::make_shared<Var>(TempName(original->var_, "src", temp_counter_++),
                                         src_slice->GetType(), original->span_);
    auto dst_slice = registry.Create("tile.slice", {original->var_, shape, offset, valid}, original->span_);
    auto dst_var = std::make_shared<Var>(TempName(original->var_, "dst", temp_counter_++),
                                         dst_slice->GetType(), original->span_);

    std::vector<ExprPtr> args{src_var, dst_var};
    if (cast->args_.size() == 2) args.push_back(cast->args_[1]);
    std::vector<std::pair<std::string, std::any>> kwargs;
    bool has_saturation = false;
    for (const auto& [key, value] : cast->kwargs_) {
      if (key == "saturation_mode") has_saturation = true;
      if (key != "target_type") kwargs.emplace_back(key, value);
    }
    if (!has_saturation) {
      if (const auto effective = GetSaturationMode(cast)) {
        kwargs.emplace_back("saturation_mode", *effective);
      }
    }
    auto fragment = registry.CreateInternal("tile.cast_fragment", args, kwargs, original->span_);
    auto fragment_var = std::make_shared<Var>(TempName(original->var_, "cast", temp_counter_++),
                                              fragment->GetType(), original->span_);
    return SeqStmts::Flatten({std::make_shared<AssignStmt>(src_var, src_slice, original->span_),
                              std::make_shared<AssignStmt>(dst_var, dst_slice, original->span_),
                              std::make_shared<AssignStmt>(fragment_var, fragment, original->span_)},
                             original->span_);
  }

  const backend::BackendHandler& handler_;
  size_t temp_counter_ = 0;
};

FunctionPtr TransformLegalizeTileCastFragments(const FunctionPtr& func) {
  if (!func || (func->level_.has_value() && *func->level_ == Level::HOST) ||
      !backend::BackendConfig::IsConfigured()) {
    return func;
  }
  const auto* context = PassContext::Current();
  const auto* handler =
      context ? context->GetBackendHandler() : backend::BackendConfig::GetBackend()->GetHandler();
  if (!handler) return func;
  LegalizeTileCastFragmentsMutator mutator(*handler);
  return mutator.VisitFunction(func);
}

}  // namespace

namespace pass {

Pass LegalizeTileCastFragments() {
  return CreateFunctionPass(TransformLegalizeTileCastFragments, "LegalizeTileCastFragments",
                            kLegalizeTileCastFragmentsProperties);
}

}  // namespace pass
}  // namespace ir
}  // namespace pypto
