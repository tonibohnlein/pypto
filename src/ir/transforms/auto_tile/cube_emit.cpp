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

#include <any>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/span.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile/cube_graph.h"
#include "src/ir/transforms/auto_tile/cube_plan.h"

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
  for (const auto& attr : attrs) {
    if (attr.first != "auto_tile") result.push_back(attr);
  }
  return result;
}

ReturnStmtPtr OriginalReturn(const FunctionPtr& function) {
  if (auto sequence = As<SeqStmts>(function->body_)) {
    for (const StmtPtr& stmt : sequence->stmts_) {
      if (auto ret = As<ReturnStmt>(stmt)) return ret;
    }
  }
  return As<ReturnStmt>(function->body_);
}

ExprPtr ClampOffset(const ExprPtr& index, int64_t region, int64_t extent, int64_t parts, const Span& span) {
  ExprPtr offset = MakeMul(index, Index(region, span), span);
  if (parts * region > extent) offset = MakeMin(offset, Index(extent - region, span), span);
  return offset;
}

ExprPtr AxisOffset(CubeAxisBinding binding, int64_t extent, const ExprPtr& m_index, const ExprPtr& n_index,
                   const Span& span) {
  switch (binding) {
    case CubeAxisBinding::Full:
      return Index(0, span);
    case CubeAxisBinding::SpatialM:
      return MakeMul(m_index, Index(extent, span), span);
    case CubeAxisBinding::SpatialN:
      return MakeMul(n_index, Index(extent, span), span);
  }
  INTERNAL_UNREACHABLE_SPAN(span) << "Internal error: unknown cube request axis binding";
}

CubeGraph RequestGraphView(const CubeGraph& graph, const CubeMatmulNode& node,
                           const CubeMatmulSchedule& request) {
  CubeGraph view = graph;
  view.m = request.output.height;
  view.n = request.output.width;
  view.k = node.k;
  view.operand_dtype = node.operand_dtype;
  view.accumulator_dtype = node.accumulator_dtype;
  view.storage_dtype = node.storage_dtype;
  return view;
}

CubeSchedulePlan RequestPlanView(const CubeSchedulePlan& plan, const CubeMatmulSchedule& request) {
  CubeSchedulePlan view = plan;
  view.output_tile_m = request.output_tile_m;
  view.output_tile_n = request.output_tile_n;
  view.output_tiles_m = request.output_tiles_m;
  view.output_tiles_n = request.output_tiles_n;
  view.k_loop = request.k_loop;
  view.retain_lhs = request.retain_lhs;
  view.retain_rhs = request.retain_rhs;
  view.l0_init = request.l0_init;
  view.l0_rolled = request.l0_rolled;
  view.l0_tail = request.l0_tail;
  return view;
}

ExprPtr MarkOutputStationary(const ExprPtr& expr, const Span& span) {
  auto call = As<Call>(expr);
  INTERNAL_CHECK_SPAN(call != nullptr, span) << "Internal error: cube matmul registry result is not a Call";
  auto attrs = call->attrs_;
  attrs.emplace_back(kCubeForceOutputStationaryAttr, true);
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                call->span_);
}

std::vector<StmtPtr> BuildMatmul(const CubeGraph& graph, const CubeSchedulePlan& plan,
                                 const ExprPtr& lhs_source, const ExprPtr& rhs_source, const ExprPtr& row,
                                 const ExprPtr& col, const ExprPtr& lhs_k_base, const ExprPtr& rhs_k_base,
                                 int64_t contraction, int64_t tile_m, int64_t tile_n, const VarPtr& result,
                                 const std::string& base, const Span& span) {
  auto& registry = OpRegistry::GetInstance();
  const auto matmul_kwargs = std::vector<std::pair<std::string, std::any>>{
      {"a_trans", false},
      {"b_trans", false},
      {"c_matrix_nz", false},
      {"out_dtype", graph.accumulator_dtype},
  };
  const auto acc_kwargs =
      std::vector<std::pair<std::string, std::any>>{{"a_trans", false}, {"b_trans", false}};

  if (!plan.k_loop.streamed()) {
    auto lhs_call = registry.Create(
        "tensor.slice", {lhs_source, IndexTuple({tile_m, contraction}, span), Pair(row, lhs_k_base, span)},
        span);
    auto lhs = std::make_shared<Var>(base + "_lhs", lhs_call->GetType(), span);
    auto rhs_call = registry.Create(
        "tensor.slice", {rhs_source, IndexTuple({contraction, tile_n}, span), Pair(rhs_k_base, col, span)},
        span);
    auto rhs = std::make_shared<Var>(base + "_rhs", rhs_call->GetType(), span);
    auto matmul =
        MarkOutputStationary(registry.Create("tensor.matmul", {lhs, rhs}, matmul_kwargs, span), span);
    return {std::make_shared<AssignStmt>(lhs, lhs_call, span),
            std::make_shared<AssignStmt>(rhs, rhs_call, span),
            std::make_shared<AssignStmt>(result, matmul, span)};
  }

  auto lhs_first_call = registry.Create(
      "tensor.slice",
      {lhs_source, IndexTuple({tile_m, plan.k_loop.chunk}, span), Pair(row, lhs_k_base, span)}, span);
  auto lhs_first = std::make_shared<Var>(base + "_lhs_first", lhs_first_call->GetType(), span);
  auto rhs_first_call = registry.Create(
      "tensor.slice",
      {rhs_source, IndexTuple({plan.k_loop.chunk, tile_n}, span), Pair(rhs_k_base, col, span)}, span);
  auto rhs_first = std::make_shared<Var>(base + "_rhs_first", rhs_first_call->GetType(), span);
  auto first_call = MarkOutputStationary(
      registry.Create("tensor.matmul", {lhs_first, rhs_first}, matmul_kwargs, span), span);
  auto first = std::make_shared<Var>(base + "_first", first_call->GetType(), span);

  auto ko = std::make_shared<Var>(base + "_ko", std::make_shared<ScalarType>(DataType::INDEX), span);
  auto carried = std::make_shared<IterArg>(base + "_carried", first->GetType(), first, span);
  const ExprPtr lhs_chunk_base = MakeAdd(lhs_k_base, ko, span);
  const ExprPtr rhs_chunk_base = MakeAdd(rhs_k_base, ko, span);

  auto lhs_call = registry.Create(
      "tensor.slice",
      {lhs_source, IndexTuple({tile_m, plan.k_loop.chunk}, span), Pair(row, lhs_chunk_base, span)}, span);
  auto lhs = std::make_shared<Var>(base + "_lhs_k", lhs_call->GetType(), span);
  auto rhs_call = registry.Create(
      "tensor.slice",
      {rhs_source, IndexTuple({plan.k_loop.chunk, tile_n}, span), Pair(rhs_chunk_base, col, span)}, span);
  auto rhs = std::make_shared<Var>(base + "_rhs_k", rhs_call->GetType(), span);

  auto rolled_call = MarkOutputStationary(
      registry.Create("tensor.matmul_acc", {ExprPtr(carried), lhs, rhs}, acc_kwargs, span), span);
  auto rolled = std::make_shared<Var>(base + "_rolled", rolled_call->GetType(), span);
  auto loop_body = SeqStmts::Flatten(
      {std::make_shared<AssignStmt>(lhs, lhs_call, span), std::make_shared<AssignStmt>(rhs, rhs_call, span),
       std::make_shared<AssignStmt>(rolled, rolled_call, span),
       std::make_shared<YieldStmt>(std::vector<ExprPtr>{rolled}, span)},
      span);

  std::vector<std::pair<std::string, std::any>> loop_attrs;
  ForKind loop_kind = ForKind::Sequential;
  const int64_t rolled_chunks = plan.k_loop.full_chunks - 1;
  if (plan.k_loop.pipeline_stages >= 2 && rolled_chunks >= 2) {
    loop_kind = ForKind::Pipeline;
    loop_attrs.emplace_back(kPipelineStagesAttr, plan.k_loop.pipeline_stages);
    loop_attrs.emplace_back(kPipelineGmToL1OnlyAttr, true);
  }
  const bool has_tail = plan.k_loop.tail > 0;
  auto loop_result = has_tail ? std::make_shared<Var>(base + "_kloop", first_call->GetType(), span) : result;
  const int64_t full_extent = plan.k_loop.full_chunks * plan.k_loop.chunk;
  auto loop =
      std::make_shared<ForStmt>(ko, Index(plan.k_loop.chunk, span), Index(full_extent, span),
                                Index(plan.k_loop.chunk, span), std::vector<IterArgPtr>{carried}, loop_body,
                                std::vector<VarPtr>{loop_result}, span, loop_kind, std::move(loop_attrs));
  std::vector<StmtPtr> statements{std::make_shared<AssignStmt>(lhs_first, lhs_first_call, span),
                                  std::make_shared<AssignStmt>(rhs_first, rhs_first_call, span),
                                  std::make_shared<AssignStmt>(first, first_call, span), loop};
  if (!has_tail) return statements;

  auto lhs_tail_call = registry.Create("tensor.slice",
                                       {lhs_source, IndexTuple({tile_m, plan.k_loop.tail}, span),
                                        Pair(row, MakeAdd(lhs_k_base, Index(full_extent, span), span), span)},
                                       span);
  auto lhs_tail = std::make_shared<Var>(base + "_lhs_tail", lhs_tail_call->GetType(), span);
  auto rhs_tail_call = registry.Create("tensor.slice",
                                       {rhs_source, IndexTuple({plan.k_loop.tail, tile_n}, span),
                                        Pair(MakeAdd(rhs_k_base, Index(full_extent, span), span), col, span)},
                                       span);
  auto rhs_tail = std::make_shared<Var>(base + "_rhs_tail", rhs_tail_call->GetType(), span);
  auto tail_call = MarkOutputStationary(
      registry.Create("tensor.matmul_acc", {loop_result, lhs_tail, rhs_tail}, acc_kwargs, span), span);
  statements.push_back(std::make_shared<AssignStmt>(lhs_tail, lhs_tail_call, span));
  statements.push_back(std::make_shared<AssignStmt>(rhs_tail, rhs_tail_call, span));
  statements.push_back(std::make_shared<AssignStmt>(result, tail_call, span));
  return statements;
}

}  // namespace

FunctionPtr EmitCubeSchedule(const CubeGraph& graph, const CubeSchedulePlan& plan,
                             const std::unordered_set<std::string>& called_functions) {
  const Span& span = graph.function->span_;
  INTERNAL_CHECK_SPAN(plan.feasible, span) << "Internal error: AutoTile cannot emit an infeasible cube plan";
  const bool geometry_valid =
      plan.parts_m > 0 && plan.parts_n > 0 && plan.region_m > 0 && plan.region_n > 0 &&
      (plan.serial_dag() || (plan.output_tile_m > 0 && plan.output_tile_n > 0 &&
                             plan.output_tiles_m * plan.output_tile_m >= plan.region_m &&
                             plan.output_tiles_n * plan.output_tile_n >= plan.region_n)) &&
      plan.spatial_work_units == plan.parts_m * plan.parts_n && plan.split_k > 0 &&
      plan.work_units == plan.spatial_work_units * plan.split_k;
  INTERNAL_CHECK_SPAN(geometry_valid, span)
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

  if (plan.serial_dag()) {
    INTERNAL_CHECK_SPAN(plan.split_k == 1 && plan.spatial_policy == CubeSpatialPolicy::Uniform &&
                            plan.matmuls.size() == plan.execution_order.size(),
                        span)
        << "Internal error: serial cube DAG requires one uniform non-split phase";
    auto work = std::make_shared<Var>(graph.function->name_ + "_work",
                                      std::make_shared<ScalarType>(DataType::INDEX), span);
    const ExprPtr m_index = MakeFloorDiv(work, Index(plan.parts_n, span), span);
    const ExprPtr n_index = MakeFloorMod(work, Index(plan.parts_n, span), span);
    std::vector<StmtPtr> kernel_body{
        std::make_shared<AssignStmt>(work, registry.Create("tile.get_block_idx", {}, span), span)};
    std::vector<ExprPtr> request_values(plan.matmuls.size());
    std::vector<ExprPtr> resident_values(plan.resident_boundaries.size());

    for (size_t order_index = 0; order_index < plan.execution_order.size(); ++order_index) {
      const size_t instance = plan.execution_order[order_index];
      INTERNAL_CHECK_SPAN(instance < plan.matmuls.size(), span)
          << "Internal error: cube request execution index is out of range";
      const CubeMatmulSchedule& request = plan.matmuls[instance];
      INTERNAL_CHECK_SPAN(request.instance == instance && request.node < graph.matmuls.size(), span)
          << "Internal error: cube request identity is inconsistent";
      const CubeMatmulNode& node = graph.matmuls[request.node];
      const std::string base = graph.function->name_ + "_request_" + std::to_string(instance);

      for (const CubeResidentBoundaryPlan& resident : plan.resident_boundaries) {
        if (resident.first_use != instance) continue;
        const ExprPtr row =
            AxisOffset(resident.region.height_binding, resident.region.height, m_index, n_index, span);
        const ExprPtr col =
            AxisOffset(resident.region.width_binding, resident.region.width, m_index, n_index, span);
        auto load = registry.Create(
            "tensor.slice",
            {resident.region.tensor, IndexTuple({resident.region.height, resident.region.width}, span),
             Pair(row, col, span)},
            span);
        auto value = std::make_shared<Var>(graph.function->name_ + "_resident_" + std::to_string(resident.id),
                                           load->GetType(), span);
        kernel_body.push_back(std::make_shared<AssignStmt>(value, load, span));
        resident_values[resident.id] = value;
      }

      auto resolve_operand = [&](const CubeTensorRegionPlan& region, int64_t producer,
                                 int64_t resident) -> std::tuple<ExprPtr, ExprPtr, ExprPtr> {
        if (producer >= 0) {
          const size_t producer_index = static_cast<size_t>(producer);
          INTERNAL_CHECK_SPAN(
              producer_index < request_values.size() && request_values[producer_index] != nullptr, span)
              << "Internal error: cube request producer is unavailable";
          return {request_values[producer_index], Index(0, span), Index(0, span)};
        }
        if (resident >= 0) {
          const size_t resident_index = static_cast<size_t>(resident);
          INTERNAL_CHECK_SPAN(
              resident_index < resident_values.size() && resident_values[resident_index] != nullptr, span)
              << "Internal error: cube resident boundary is unavailable";
          return {resident_values[resident_index], Index(0, span), Index(0, span)};
        }
        return {region.tensor, AxisOffset(region.height_binding, region.height, m_index, n_index, span),
                AxisOffset(region.width_binding, region.width, m_index, n_index, span)};
      };

      auto [lhs_source, lhs_row, lhs_k] =
          resolve_operand(request.lhs, request.lhs_producer, request.lhs_resident_boundary);
      auto [rhs_source, rhs_k, rhs_col] =
          resolve_operand(request.rhs, request.rhs_producer, request.rhs_resident_boundary);
      if (request.retain_lhs) {
        auto preload = registry.Create("tensor.slice",
                                       {lhs_source, IndexTuple({request.lhs.height, request.lhs.width}, span),
                                        Pair(lhs_row, lhs_k, span)},
                                       span);
        auto retained = std::make_shared<Var>(base + "_lhs_l1", preload->GetType(), span);
        kernel_body.push_back(std::make_shared<AssignStmt>(retained, preload, span));
        lhs_source = retained;
        lhs_row = Index(0, span);
        lhs_k = Index(0, span);
      }
      if (request.retain_rhs) {
        auto preload = registry.Create("tensor.slice",
                                       {rhs_source, IndexTuple({request.rhs.height, request.rhs.width}, span),
                                        Pair(rhs_k, rhs_col, span)},
                                       span);
        auto retained = std::make_shared<Var>(base + "_rhs_l1", preload->GetType(), span);
        kernel_body.push_back(std::make_shared<AssignStmt>(retained, preload, span));
        rhs_source = retained;
        rhs_k = Index(0, span);
        rhs_col = Index(0, span);
      }

      ExprPtr output_state = output_buffer;
      if (!request.is_sink) {
        auto create = registry.Create("tensor.create_l1",
                                      {IndexTuple({request.output.height, request.output.width}, span)},
                                      {{"dtype", node.storage_dtype}, {"transpose", false}}, span);
        auto scratch = std::make_shared<Var>(base + "_l1", create->GetType(), span);
        kernel_body.push_back(std::make_shared<AssignStmt>(scratch, create, span));
        output_state = scratch;
      }

      const int64_t tile_count = request.output_tiles_m * request.output_tiles_n;
      auto tile_index =
          std::make_shared<Var>(base + "_tile", std::make_shared<ScalarType>(DataType::INDEX), span);
      const ExprPtr tile_m_index = MakeFloorDiv(tile_index, Index(request.output_tiles_n, span), span);
      const ExprPtr tile_n_index = MakeFloorMod(tile_index, Index(request.output_tiles_n, span), span);
      const ExprPtr local_row = ClampOffset(tile_m_index, request.output_tile_m, request.output.height,
                                            request.output_tiles_m, span);
      const ExprPtr local_col = ClampOffset(tile_n_index, request.output_tile_n, request.output.width,
                                            request.output_tiles_n, span);
      auto root = std::make_shared<IterArg>(base + "_root", output_state->GetType(), output_state, span);
      auto acc = std::make_shared<Var>(
          base + "_acc",
          std::make_shared<TensorType>(
              std::vector<ExprPtr>{Index(request.output_tile_m, span), Index(request.output_tile_n, span)},
              node.accumulator_dtype),
          span);
      const CubeGraph request_graph = RequestGraphView(graph, node, request);
      const CubeSchedulePlan request_plan = RequestPlanView(plan, request);
      std::vector<StmtPtr> tile_body;
      for (StmtPtr& statement :
           BuildMatmul(request_graph, request_plan, lhs_source, rhs_source, MakeAdd(lhs_row, local_row, span),
                       MakeAdd(rhs_col, local_col, span), lhs_k, rhs_k, node.k, request.output_tile_m,
                       request.output_tile_n, acc, base + "_tile_body", span)) {
        tile_body.push_back(std::move(statement));
      }
      const ExprPtr output_row = request.is_sink
                                     ? MakeAdd(AxisOffset(request.output.height_binding,
                                                          request.output.height, m_index, n_index, span),
                                               local_row, span)
                                     : local_row;
      const ExprPtr output_col = request.is_sink
                                     ? MakeAdd(AxisOffset(request.output.width_binding, request.output.width,
                                                          m_index, n_index, span),
                                               local_col, span)
                                     : local_col;
      auto assemble =
          registry.Create("tensor.assemble", {ExprPtr(root), acc, Pair(output_row, output_col, span)}, span);
      auto assembled = std::make_shared<Var>(base + "_assembled", assemble->GetType(), span);
      tile_body.push_back(std::make_shared<AssignStmt>(assembled, assemble, span));
      tile_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{assembled}, span));

      VarPtr result =
          request.is_sink ? graph.output : std::make_shared<Var>(base + "_out", assemble->GetType(), span);
      auto tile_loop = std::make_shared<ForStmt>(
          tile_index, Index(0, span), Index(tile_count, span), Index(1, span), std::vector<IterArgPtr>{root},
          SeqStmts::Flatten(std::move(tile_body), span), std::vector<VarPtr>{result}, span,
          ForKind::Sequential, std::vector<std::pair<std::string, std::any>>{});
      kernel_body.push_back(tile_loop);
      request_values[instance] = result;
    }

    auto kernel = std::make_shared<InCoreScopeStmt>(SplitMode::None, graph.function->name_,
                                                    SeqStmts::Flatten(std::move(kernel_body), span), span);
    body.push_back(std::make_shared<SpmdScopeStmt>(Index(plan.spatial_work_units, span), false,
                                                   graph.function->name_ + "_spmd", kernel, span));
    body.push_back(OriginalReturn(graph.function));

    std::vector<VarPtr> params = graph.function->params_;
    std::vector<ParamDirection> directions = graph.function->param_directions_;
    if (lift_output) {
      params.push_back(output_buffer);
      directions.push_back(ParamDirection::Out);
    }
    return std::make_shared<Function>(
        graph.function->name_, std::move(params), std::move(directions), graph.function->return_types_,
        SeqStmts::Flatten(std::move(body), span), graph.function->span_, graph.function->func_type_,
        graph.function->level_, graph.function->role_, WithoutAutoTile(graph.function->attrs_),
        graph.function->requires_runtime_binding_);
  }

  const int64_t effective_k = graph.k / plan.split_k;
  auto emit_phase = [&](const std::string& phase_name, int64_t work_units, int64_t split_offset,
                        int64_t split_count, bool atomic, bool bind_output,
                        const ExprPtr& initial_output) -> VarPtr {
    INTERNAL_CHECK_SPAN(work_units > 0 && split_count > 0, span)
        << "Internal error: AutoTile cube split phase has no work";
    auto work =
        std::make_shared<Var>(phase_name + "_work", std::make_shared<ScalarType>(DataType::INDEX), span);
    ExprPtr split_index = Index(split_offset, span);
    ExprPtr spatial_index = work;
    if (split_count > 1) {
      split_index =
          MakeAdd(MakeFloorMod(work, Index(split_count, span), span), Index(split_offset, span), span);
      spatial_index = MakeFloorDiv(work, Index(split_count, span), span);
    }
    const ExprPtr m_index = MakeFloorDiv(spatial_index, Index(plan.parts_n, span), span);
    const ExprPtr n_index = MakeFloorMod(spatial_index, Index(plan.parts_n, span), span);
    const ExprPtr row = ClampOffset(m_index, plan.region_m, graph.m, plan.parts_m, span);
    const ExprPtr col = ClampOffset(n_index, plan.region_n, graph.n, plan.parts_n, span);
    const ExprPtr k_base = MakeMul(split_index, Index(effective_k, span), span);

    std::vector<StmtPtr> kernel_body;
    kernel_body.push_back(
        std::make_shared<AssignStmt>(work, registry.Create("tile.get_block_idx", {}, span), span));

    ExprPtr lhs_source = graph.lhs;
    ExprPtr rhs_source = graph.rhs;
    ExprPtr lhs_row = row;
    ExprPtr lhs_k = k_base;
    ExprPtr rhs_k = k_base;
    ExprPtr rhs_col = col;
    if (plan.retain_lhs) {
      auto retained_call = registry.Create(
          "tensor.slice",
          {lhs_source, IndexTuple({plan.region_m, effective_k}, span), Pair(row, k_base, span)}, span);
      auto retained = std::make_shared<Var>(phase_name + "_lhs_l1", retained_call->GetType(), span);
      kernel_body.push_back(std::make_shared<AssignStmt>(retained, retained_call, span));
      lhs_source = retained;
      lhs_row = Index(0, span);
      lhs_k = Index(0, span);
    }
    if (plan.retain_rhs) {
      auto retained_call = registry.Create(
          "tensor.slice",
          {rhs_source, IndexTuple({effective_k, plan.region_n}, span), Pair(k_base, col, span)}, span);
      auto retained = std::make_shared<Var>(phase_name + "_rhs_l1", retained_call->GetType(), span);
      kernel_body.push_back(std::make_shared<AssignStmt>(retained, retained_call, span));
      rhs_source = retained;
      rhs_k = Index(0, span);
      rhs_col = Index(0, span);
    }

    const int64_t tile_count = plan.output_tiles_m * plan.output_tiles_n;
    auto tile_index =
        std::make_shared<Var>(phase_name + "_tile", std::make_shared<ScalarType>(DataType::INDEX), span);
    const ExprPtr tile_m_index = MakeFloorDiv(tile_index, Index(plan.output_tiles_n, span), span);
    const ExprPtr tile_n_index = MakeFloorMod(tile_index, Index(plan.output_tiles_n, span), span);
    const ExprPtr local_row =
        ClampOffset(tile_m_index, plan.output_tile_m, plan.region_m, plan.output_tiles_m, span);
    const ExprPtr local_col =
        ClampOffset(tile_n_index, plan.output_tile_n, plan.region_n, plan.output_tiles_n, span);
    auto root =
        std::make_shared<IterArg>(phase_name + "_root", initial_output->GetType(), initial_output, span);
    const std::string tile_base = phase_name + "_tile_body";
    auto tile = std::make_shared<Var>(
        tile_base + "_acc",
        std::make_shared<TensorType>(
            std::vector<ExprPtr>{Index(plan.output_tile_m, span), Index(plan.output_tile_n, span)},
            graph.accumulator_dtype),
        span);
    std::vector<StmtPtr> tile_body;
    for (StmtPtr& statement :
         BuildMatmul(graph, plan, lhs_source, rhs_source, MakeAdd(lhs_row, local_row, span),
                     MakeAdd(rhs_col, local_col, span), lhs_k, rhs_k, effective_k, plan.output_tile_m,
                     plan.output_tile_n, tile, tile_base, span)) {
      tile_body.push_back(std::move(statement));
    }
    std::vector<std::pair<std::string, std::any>> kwargs;
    if (atomic) kwargs.emplace_back("atomic", 1);
    auto assemble = registry.Create(
        "tensor.assemble",
        {ExprPtr(root), tile, Pair(MakeAdd(row, local_row, span), MakeAdd(col, local_col, span), span)},
        kwargs, span);
    auto tile_state = std::make_shared<Var>(tile_base + "_gm", assemble->GetType(), span);
    tile_body.push_back(std::make_shared<AssignStmt>(tile_state, assemble, span));
    tile_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{tile_state}, span));

    VarPtr next =
        bind_output ? graph.output : std::make_shared<Var>(phase_name + "_out", assemble->GetType(), span);
    auto tile_loop = std::make_shared<ForStmt>(
        tile_index, Index(0, span), Index(tile_count, span), Index(1, span), std::vector<IterArgPtr>{root},
        SeqStmts::Flatten(std::move(tile_body), span), std::vector<VarPtr>{next}, span, ForKind::Sequential,
        std::vector<std::pair<std::string, std::any>>{});
    kernel_body.push_back(tile_loop);

    auto kernel = std::make_shared<InCoreScopeStmt>(SplitMode::None, phase_name,
                                                    SeqStmts::Flatten(std::move(kernel_body), span), span);
    body.push_back(
        std::make_shared<SpmdScopeStmt>(Index(work_units, span), false, phase_name + "_spmd", kernel, span));
    return next;
  };

  if (plan.first_partial_then_atomic()) {
    VarPtr first = emit_phase(graph.function->name_ + "_first_partial", plan.first_partial_work_units,
                              /*split_offset=*/0,
                              /*split_count=*/1, /*atomic=*/false,
                              /*bind_output=*/false, output_buffer);
    // Thread the post-call SSA result into the atomic phase. InOutUseDiscipline
    // requires this representation; outlining and orchestration codegen retain
    // the underlying OUTPUT_EXISTING/InOut buffer identity and establish the
    // ordered write dependency between the two launches.
    (void)emit_phase(graph.function->name_ + "_atomic_rest", plan.atomic_rest_work_units,
                     /*split_offset=*/1, /*split_count=*/plan.split_k - 1,
                     /*atomic=*/true, /*bind_output=*/true, first);
  } else {
    (void)emit_phase(graph.function->name_, plan.spatial_work_units, /*split_offset=*/0,
                     /*split_count=*/1, /*atomic=*/false, /*bind_output=*/true, output_buffer);
  }
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
