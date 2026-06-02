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

// AutoFuse: automatic operator fusion + tile-size selection on the tensor graph.
//
// The pass extracts the tensor-op DAG from a (static-shape, tensor-level)
// function, hands it to the MLSys graph-scheduling solver (linked as
// `solver_lib` from 3rdparty/mlsys26) to choose a memory-reuse partition and
// tile granularity, and then materialises the chosen grouping as nested
// InCoreScopeStmt regions for OutlineIncoreScopes to lift into kernels.
//
// v0 status: builds the solver `Problem` from the IR and runs `solve()`. The
// IR rewrite (emit InCoreScopeStmt from the returned schedule) is the next
// increment — see TODO in AutoFuseTransform.

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/type.h"

// MLSys graph-scheduling solver (3rdparty/mlsys26), linked as `solver_lib`.
#include "core/dag.h"
#include "core/types.h"
#include "pipeline/solver.h"
#include "solution/solution.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

// Placeholder compute-cost model. TODO: ground in BackendHandler throughput.
int64_t ComputeCost(::OpType type, int64_t w, int64_t h, int64_t k) {
  // PLACEHOLDER work/throughput model. The throughput is tuned so a pointwise op
  // is memory-bound (compute < DDR transfer) like the real vector unit —
  // otherwise fusion, which only saves memory traffic, shows no benefit.
  // TODO(cost-model): ground in BackendHandler throughput.
  constexpr int64_t kThroughput = 64;
  if (type == ::OpType::MatMul) {
    return (w * h * (k > 0 ? k : w)) / kThroughput;
  }
  return (w * h) / kThroughput;
}

// Hardware parameters (placeholders). TODO: read from the active BackendHandler.
constexpr int64_t kFastMemoryCapacity = 50000;   // UB capacity, element-count convention
constexpr int64_t kSlowMemoryBandwidth = 10;     // DDR bandwidth
constexpr int64_t kNativeW = 128;
constexpr int64_t kNativeH = 128;

// Walk a tensor-level function body and build the solver Problem.
class ProblemBuilder : public IRVisitor {
 public:
  ::Problem problem;

  void Build(const FunctionPtr& func) {
    problem.fast_memory_capacity = kFastMemoryCapacity;
    problem.slow_memory_bandwidth = kSlowMemoryBandwidth;
    problem.native_w = kNativeW;
    problem.native_h = kNativeH;
    for (const auto& param : func->params_) {
      TensorId(param);
    }
    VisitStmt(func->body_);
  }

  void VisitStmt_(const AssignStmtPtr& op) override {
    auto call = As<Call>(op->value_);
    if (call != nullptr) {
      ::Op sop;
      const std::string& name = call->op_->name_;
      sop.type = (name.find("matmul") != std::string::npos) ? ::OpType::MatMul : ::OpType::Pointwise;
      for (const auto& arg : call->args_) {
        auto var = AsVarLike(arg);
        if (var != nullptr) {
          sop.inputs.push_back(TensorId(var));
        }
      }
      const size_t out = TensorId(op->var_);
      sop.outputs.push_back(out);
      const ::Tensor& out_t = problem.tensors[out];
      sop.base_cost = ComputeCost(sop.type, out_t.width, out_t.height, out_t.width);
      problem.ops.push_back(std::move(sop));
    }
    // Do not descend into the value expression — ops are AssignStmt-granular here.
  }

 private:
  std::unordered_map<const Var*, size_t> tensor_index_;

  size_t TensorId(const VarPtr& var) {
    const Var* raw = var.get();
    auto it = tensor_index_.find(raw);
    if (it != tensor_index_.end()) {
      return it->second;
    }
    auto tt = As<TensorType>(var->GetType());
    CHECK(tt != nullptr) << "AutoFuse: variable '" << var->name_hint_
                         << "' is not tensor-typed (only static tensor-level functions are supported)";
    int64_t w = 0;
    int64_t h = 0;
    ShapeWH(tt, &w, &h);
    const size_t idx = problem.tensors.size();
    problem.tensors.push_back(::Tensor{w, h});
    tensor_index_[raw] = idx;
    return idx;
  }

  static void ShapeWH(const TensorTypePtr& tt, int64_t* w, int64_t* h) {
    const auto& shape = tt->shape_;
    auto dim = [&](size_t i) -> int64_t {
      auto ci = As<ConstInt>(shape[i]);
      CHECK(ci != nullptr) << "AutoFuse: dynamic/symbolic tensor shapes are out of scope for v0";
      return ci->value_;
    };
    if (shape.size() >= 2) {
      *h = dim(shape.size() - 2);
      *w = dim(shape.size() - 1);
    } else if (shape.size() == 1) {
      *w = dim(0);
      *h = 1;
    } else {
      *w = 1;
      *h = 1;
    }
  }
};

ProgramPtr AutoFuseTransform(const ProgramPtr& prog) {
  for (const auto& entry : prog->functions_) {
    const FunctionPtr& func = entry.second;
    // v0 gate: only functions explicitly marked for auto-fusion. attrs_ is an
    // ordered vector of (key, value) pairs, not a map.
    const bool marked = std::any_of(func->attrs_.begin(), func->attrs_.end(),
                                    [](const auto& kv) { return kv.first == "auto_fuse"; });
    if (!marked) {
      continue;
    }
    ProblemBuilder builder;
    builder.Build(func);
    if (builder.problem.ops.empty()) {
      continue;
    }
    ::DAG dag = ::DAG::build(builder.problem);
    ::Solution sol = ::solve(builder.problem, dag);
    LOG_INFO << "AutoFuse[" << func->name_ << "]: " << builder.problem.ops.size() << " ops -> "
             << sol.num_steps() << " fused subgraph(s), total latency " << sol.total_latency();
    // TODO(next increment): rewrite `func` — emit one InCoreScopeStmt per
    // sol.step(i).subgraph, with the chosen tile config, in a valid topological
    // order, for OutlineIncoreScopes to lift into kernels.
  }
  return prog;
}

}  // namespace

Pass AutoFuse() { return CreateProgramPass(AutoFuseTransform, "AutoFuse", {}); }

}  // namespace pass
}  // namespace ir
}  // namespace pypto
