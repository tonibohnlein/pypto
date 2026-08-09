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

#include "src/ir/transforms/auto_tile/cube_emit.h"

#include <algorithm>
#include <any>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/error.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

ExprPtr Index(int64_t value, const Span& span) {
  return std::make_shared<ConstInt>(value, DataType::INDEX, span);
}

ExprPtr IndexTuple(std::initializer_list<int64_t> values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (int64_t value : values) elements.push_back(Index(value, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

ExprPtr Pair(const ExprPtr& lhs, const ExprPtr& rhs, const Span& span) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{lhs, rhs}, span);
}

std::vector<std::pair<std::string, std::any>> WithoutAutoTile(
    const std::vector<std::pair<std::string, std::any>>& attrs) {
  std::vector<std::pair<std::string, std::any>> result;
  result.reserve(attrs.size());
  for (const auto& attr : attrs)
    if (attr.first != "auto_tile") result.push_back(attr);
  return result;
}

ReturnStmtPtr OriginalReturn(const FunctionPtr& function) {
  if (auto sequence = As<SeqStmts>(function->body_)) {
    for (const StmtPtr& stmt : sequence->stmts_)
      if (auto ret = As<ReturnStmt>(stmt)) return ret;
  }
  return As<ReturnStmt>(function->body_);
}

ExprPtr ClampOffset(const ExprPtr& index, int64_t region, int64_t extent, int64_t parts, const Span& span) {
  ExprPtr offset = MakeMul(index, Index(region, span), span);
  if (parts * region > extent) offset = MakeMin(offset, Index(extent - region, span), span);
  return offset;
}

}  // namespace

FunctionPtr EmitCubeSchedule(const CubeGraph& graph, const CubeSchedulePlan& plan,
                             const std::unordered_set<std::string>& called_functions) {
  const Span& span = graph.function->span_;
  INTERNAL_CHECK_SPAN(plan.feasible, span) << "Internal error: AutoTile cannot emit an infeasible cube plan";
  INTERNAL_CHECK_SPAN(plan.parts_m > 0 && plan.parts_n > 0 && plan.region_m > 0 && plan.region_n > 0 &&
                          plan.work_units == plan.parts_m * plan.parts_n,
                      span)
      << "Internal error: AutoTile cube plan has an invalid spatial grid";
  INTERNAL_CHECK_SPAN(plan.region_m <= graph.m && plan.region_n <= graph.n &&
                          plan.parts_m * plan.region_m >= graph.m && plan.parts_n * plan.region_n >= graph.n,
                      span)
      << "Internal error: AutoTile cube plan does not cover its output";

  auto& registry = OpRegistry::GetInstance();
  const bool lift_output =
      graph.explicit_output_buffer == nullptr && called_functions.count(graph.function->name_) == 0;
  VarPtr output_buffer = graph.explicit_output_buffer;
  std::vector<StmtPtr> body;
  if (output_buffer == nullptr) {
    auto create = registry.Create("tensor.create", {IndexTuple({graph.m, graph.n}, span)},
                                  {{"dtype", graph.storage_dtype}, {"layout", TensorLayout::ND}}, span);
    output_buffer = std::make_shared<Var>(graph.output->name_hint_ + "_out", create->GetType(), span);
    if (!lift_output) body.push_back(std::make_shared<AssignStmt>(output_buffer, create, span));
  }

  auto work =
      std::make_shared<Var>("__auto_tile_cube_region", std::make_shared<ScalarType>(DataType::INDEX), span);
  const ExprPtr m_index = MakeFloorDiv(work, Index(plan.parts_n, span), span);
  const ExprPtr n_index = MakeFloorMod(work, Index(plan.parts_n, span), span);
  const ExprPtr row = ClampOffset(m_index, plan.region_m, graph.m, plan.parts_m, span);
  const ExprPtr col = ClampOffset(n_index, plan.region_n, graph.n, plan.parts_n, span);

  std::vector<StmtPtr> kernel_body;
  kernel_body.push_back(
      std::make_shared<AssignStmt>(work, registry.Create("tile.get_block_idx", {}, span), span));
  auto lhs_slice = registry.Create(
      "tensor.slice",
      {graph.lhs, IndexTuple({plan.region_m, graph.k}, span), Pair(row, Index(0, span), span)}, span);
  auto lhs = std::make_shared<Var>("__auto_tile_cube_lhs", lhs_slice->GetType(), span);
  kernel_body.push_back(std::make_shared<AssignStmt>(lhs, lhs_slice, span));
  auto rhs_slice = registry.Create(
      "tensor.slice",
      {graph.rhs, IndexTuple({graph.k, plan.region_n}, span), Pair(Index(0, span), col, span)}, span);
  auto rhs = std::make_shared<Var>("__auto_tile_cube_rhs", rhs_slice->GetType(), span);
  kernel_body.push_back(std::make_shared<AssignStmt>(rhs, rhs_slice, span));

  auto matmul = registry.Create("tensor.matmul", {lhs, rhs},
                                {{"a_trans", false},
                                 {"b_trans", false},
                                 {"c_matrix_nz", false},
                                 {"out_dtype", graph.accumulator_dtype}},
                                span);
  auto tile = std::make_shared<Var>("__auto_tile_cube_acc", matmul->GetType(), span);
  kernel_body.push_back(std::make_shared<AssignStmt>(tile, matmul, span));
  auto assemble = registry.Create("tensor.assemble", {output_buffer, tile, Pair(row, col, span)}, span);
  kernel_body.push_back(std::make_shared<AssignStmt>(graph.output, assemble, span));

  auto kernel = std::make_shared<InCoreScopeStmt>(SplitMode::None, graph.function->name_,
                                                  SeqStmts::Flatten(std::move(kernel_body), span), span);
  body.push_back(std::make_shared<SpmdScopeStmt>(Index(plan.work_units, span), false,
                                                 graph.function->name_ + "_spmd", kernel, span));
  body.push_back(OriginalReturn(graph.function));

  std::vector<VarPtr> params = graph.function->params_;
  std::vector<ParamDirection> directions = graph.function->param_directions_;
  if (lift_output) {
    params.push_back(output_buffer);
    directions.push_back(ParamDirection::Out);
  }
  return std::make_shared<Function>(graph.function->name_, std::move(params), std::move(directions),
                                    graph.function->return_types_, SeqStmts::Flatten(std::move(body), span),
                                    graph.function->span_, graph.function->func_type_, graph.function->level_,
                                    graph.function->role_, WithoutAutoTile(graph.function->attrs_),
                                    graph.function->requires_runtime_binding_);
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
