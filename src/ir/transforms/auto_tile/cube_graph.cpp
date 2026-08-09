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

#include "src/ir/transforms/auto_tile/cube_graph.h"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/type.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

bool IsCubeOp(const CallPtr& call) {
  return call != nullptr && (IsOp(call, "tensor.matmul") || IsOp(call, "tensor.matmul_acc"));
}

bool IsSupportedOperandDType(const DataType& dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

bool IsSupportedStorageDType(const DataType& dtype) {
  return dtype == DataType::FP16 || dtype == DataType::BF16 || dtype == DataType::FP32;
}

std::pair<int64_t, int64_t> StaticRank2TensorShape(const TypePtr& type) {
  auto tensor = As<TensorType>(type);
  if (tensor == nullptr || tensor->shape_.size() != 2) return {-1, -1};
  auto rows = As<ConstInt>(tensor->shape_[0]);
  auto cols = As<ConstInt>(tensor->shape_[1]);
  if (rows == nullptr || cols == nullptr) return {-1, -1};
  return {rows->value_, cols->value_};
}

std::vector<StmtPtr> TopLevelStatements(const FunctionPtr& function) {
  if (auto sequence = As<SeqStmts>(function->body_)) return sequence->stmts_;
  return {function->body_};
}

CubeGraph BuildCubeGraphOrThrow(const FunctionPtr& function, const ProgramPtr& program) {
  (void)program;
  CHECK_SPAN(function != nullptr && function->body_ != nullptr, function ? function->span_ : Span::unknown())
      << "AutoTile requires a function body";

  CubeGraph graph;
  graph.function = function;
  ReturnStmtPtr return_stmt;
  std::unordered_map<const Var*, size_t> producer_by_value;
  for (const StmtPtr& stmt : TopLevelStatements(function)) {
    if (auto ret = As<ReturnStmt>(stmt)) {
      CHECK_SPAN(return_stmt == nullptr, ret->span_) << "AutoTile requires one top-level return";
      return_stmt = ret;
      continue;
    }
    auto assign = As<AssignStmt>(stmt);
    CHECK_SPAN(assign != nullptr, stmt->span_)
        << "AutoTile cube scheduling requires a straight-line tensor DAG";
    auto call = As<Call>(assign->value_);
    CHECK_SPAN(call != nullptr && call->op_ != nullptr, assign->span_)
        << "AutoTile cube scheduling requires every tensor definition to be a direct operation call";
    CHECK_SPAN(IsOp(call, "tensor.matmul"), call->span_)
        << "AutoTile cannot form one homogeneous cube schedule because operation '" << call->op_->name_
        << "' is not a supported tensor.matmul";
    CHECK_SPAN(call->args_.size() == 2, call->span_)
        << "AutoTile tensor.matmul requires exactly two operands";
    CHECK_SPAN(!call->GetKwarg<bool>("a_trans", false) && !call->GetKwarg<bool>("b_trans", false) &&
                   !call->GetKwarg<bool>("c_matrix_nz", false),
               call->span_)
        << "AutoTile cube scheduling currently supports non-transposed ND tensor.matmul only";

    CubeMatmulNode node;
    node.stmt = assign;
    node.call = call;
    node.lhs = AsVarLike(call->args_[0]);
    node.rhs = AsVarLike(call->args_[1]);
    node.output = assign->var_;
    CHECK_SPAN(node.lhs != nullptr && node.rhs != nullptr, call->span_)
        << "AutoTile tensor.matmul operands must be named SSA values after FlattenCallExpr";
    if (auto found = producer_by_value.find(node.lhs.get()); found != producer_by_value.end()) {
      node.lhs_producer = static_cast<int64_t>(found->second);
    }
    if (auto found = producer_by_value.find(node.rhs.get()); found != producer_by_value.end()) {
      node.rhs_producer = static_cast<int64_t>(found->second);
    }

    auto lhs_type = As<TensorType>(node.lhs->GetType());
    auto rhs_type = As<TensorType>(node.rhs->GetType());
    auto output_type = As<TensorType>(node.output->GetType());
    CHECK_SPAN(lhs_type != nullptr && rhs_type != nullptr && output_type != nullptr, call->span_)
        << "AutoTile cube scheduling supports Tensor operands and results only";
    const auto [lhs_m, lhs_k] = StaticRank2TensorShape(lhs_type);
    const auto [rhs_k, rhs_n] = StaticRank2TensorShape(rhs_type);
    const auto [output_m, output_n] = StaticRank2TensorShape(output_type);
    CHECK_SPAN(lhs_m > 0 && lhs_k > 0 && rhs_k > 0 && rhs_n > 0 && output_m > 0 && output_n > 0, call->span_)
        << "AutoTile cube scheduling requires positive, static rank-2 tensors";
    CHECK_SPAN(lhs_k == rhs_k && output_m == lhs_m && output_n == rhs_n, call->span_)
        << "AutoTile tensor.matmul shapes do not satisfy [M,K] @ [K,N] -> [M,N]";
    CHECK_SPAN(lhs_type->dtype_ == rhs_type->dtype_, call->span_)
        << "AutoTile cube scheduling requires equal tensor.matmul operand dtypes";
    CHECK_SPAN(IsSupportedOperandDType(lhs_type->dtype_), call->span_)
        << "AutoTile Ascend910B cube scheduling supports FP16, BF16, and FP32 operands";
    CHECK_SPAN(IsSupportedStorageDType(output_type->dtype_), call->span_)
        << "AutoTile Ascend910B cube scheduling supports FP16, BF16, and FP32 result storage";

    node.m = lhs_m;
    node.n = rhs_n;
    node.k = lhs_k;
    node.operand_dtype = lhs_type->dtype_;
    node.accumulator_dtype = lhs_type->dtype_.IsFloat() ? DataType::FP32 : DataType::INT32;
    node.storage_dtype = output_type->dtype_;
    const size_t index = graph.matmuls.size();
    graph.matmuls.push_back(std::move(node));
    producer_by_value.emplace(assign->var_.get(), index);
  }

  CHECK_SPAN(!graph.matmuls.empty(), function->span_)
      << "AutoTile cube scheduling requires at least one tensor.matmul";
  CHECK_SPAN(return_stmt != nullptr && return_stmt->value_.size() == 1, function->span_)
      << "AutoTile cube scheduling requires exactly one returned tensor";
  auto returned = AsVarLike(return_stmt->value_.front());
  auto sink_it = returned != nullptr ? producer_by_value.find(returned.get()) : producer_by_value.end();
  CHECK_SPAN(sink_it != producer_by_value.end(), return_stmt->span_)
      << "AutoTile cube scheduling requires the returned tensor to be produced by tensor.matmul";
  graph.sink = sink_it->second;
  graph.matmuls[graph.sink].is_sink = true;

  std::unordered_set<size_t> reachable;
  auto mark_reachable = [&](auto&& self, size_t index) -> void {
    if (!reachable.insert(index).second) return;
    const CubeMatmulNode& node = graph.matmuls[index];
    if (node.lhs_producer >= 0) self(self, static_cast<size_t>(node.lhs_producer));
    if (node.rhs_producer >= 0) self(self, static_cast<size_t>(node.rhs_producer));
  };
  mark_reachable(mark_reachable, graph.sink);
  CHECK_SPAN(reachable.size() == graph.matmuls.size(), function->span_)
      << "AutoTile cube scheduling requires every matmul to contribute to the returned tensor";

  std::unordered_map<const Var*, ParamDirection> parameter_directions;
  for (size_t i = 0; i < function->params_.size(); ++i) {
    const ParamDirection direction =
        i < function->param_directions_.size() ? function->param_directions_[i] : ParamDirection::In;
    parameter_directions.emplace(function->params_[i].get(), direction);
    if (direction == ParamDirection::Out) {
      CHECK_SPAN(graph.explicit_output_buffer == nullptr, function->params_[i]->span_)
          << "AutoTile cube scheduling supports exactly one explicit Out tensor";
      graph.explicit_output_buffer = function->params_[i];
    }
  }
  for (const CubeMatmulNode& node : graph.matmuls) {
    for (const auto& [operand, producer] : {std::pair<VarPtr, int64_t>{node.lhs, node.lhs_producer},
                                            std::pair<VarPtr, int64_t>{node.rhs, node.rhs_producer}}) {
      if (producer >= 0) continue;
      auto found = parameter_directions.find(operand.get());
      CHECK_SPAN(found != parameter_directions.end() &&
                     (found->second == ParamDirection::In || found->second == ParamDirection::InOut),
                 operand->span_)
          << "AutoTile cube operand '" << operand->name_hint_ << "' is not a readable function parameter";
    }
    if (!node.is_sink) {
      CHECK_SPAN(node.storage_dtype == DataType::FP16 || node.storage_dtype == DataType::BF16,
                 node.stmt->span_)
          << "AutoTile cube internal matmul '" << node.output->name_hint_
          << "' must use FP16 or BF16 storage for the Acc-to-Mat handoff";
    }
  }

  const CubeMatmulNode& sink = graph.matmuls[graph.sink];
  graph.matmul_stmt = sink.stmt;
  graph.matmul_call = sink.call;
  graph.lhs = sink.lhs;
  graph.rhs = sink.rhs;
  graph.output = sink.output;
  graph.m = sink.m;
  graph.n = sink.n;
  graph.k = sink.k;
  graph.operand_dtype = sink.operand_dtype;
  graph.accumulator_dtype = sink.accumulator_dtype;
  graph.storage_dtype = sink.storage_dtype;
  if (graph.explicit_output_buffer != nullptr) {
    CHECK_SPAN(structural_equal(graph.explicit_output_buffer->GetType(), graph.output->GetType()),
               graph.explicit_output_buffer->span_)
        << "AutoTile returned tensor is not type-compatible with explicit Out parameter '"
        << graph.explicit_output_buffer->name_hint_ << "'";
  }
  return graph;
}

}  // namespace

bool ContainsCubeOperation(const FunctionPtr& function) {
  if (function == nullptr || function->body_ == nullptr) return false;
  for (const StmtPtr& stmt : TopLevelStatements(function)) {
    auto assign = As<AssignStmt>(stmt);
    if (assign != nullptr && IsCubeOp(As<Call>(assign->value_))) return true;
  }
  return false;
}

CubeAdmissionResult AdmitCubeGraph(const FunctionPtr& function, const ProgramPtr& program) {
  CubeAdmissionResult result;
  try {
    result.graph = BuildCubeGraphOrThrow(function, program);
    result.supported = true;
  } catch (const pypto::Error& error) {
    result.reason = error.what();
    result.failure = std::current_exception();
  }
  return result;
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
