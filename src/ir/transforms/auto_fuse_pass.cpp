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

// AutoFuse: automatic operator fusion + tile-size selection.
//
// The extractor builds the MLSys solver's op+tensor DAG (`Problem`) from an
// `auto_fuse`-marked function by reusing PyPTO's own dependency analysis
// (`BuildStmtDependencyGraph`), which is Out/InOut/SSA-correct. This handles
// both forms uniformly:
//   * a flat tensor-level function (each AssignStmt is a tensor op), and
//   * an orchestration kernel-call DAG (`c_v1 = self.kernel_add(a, b, c)`),
//     where `tensor.create` allocations and Out-buffer args are skipped.
// The DAG is handed to the linked MLSys solver (`3rdparty/mlsys26`) to choose a
// memory-reuse fusion partition. v0 computes + logs (and optionally dumps) the
// grouping; the IR rewrite (emit InCoreScopeStmt) is the next increment.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/soc.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/type.h"

// MLSys graph-scheduling solver (3rdparty/mlsys26), linked as `solver_lib`.
#include "core/dag.h"
#include "core/subgraph.h"
#include "core/types.h"
#include "pipeline/solver.h"
#include "solution/solution.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

// Placeholder work/throughput cost model. The throughput is tuned so a pointwise
// op is memory-bound (compute < DDR transfer) like the real vector unit —
// otherwise fusion, which only saves memory traffic, shows no benefit.
// TODO(cost-model): ground in BackendHandler throughput.
int64_t ComputeCost(::OpType type, int64_t w, int64_t h, int64_t k) {
  constexpr int64_t kThroughput = 64;
  if (type == ::OpType::MatMul) {
    return (w * h * (k > 0 ? k : w)) / kThroughput;
  }
  return (w * h) / kThroughput;
}

// Hardware parameters. v0 hardcodes the Ascend 910B machine model (mirrors
// `set_910b` in 3rdparty/mlsys26/test/ascend_910b_test.cpp); with the
// tile-geometry compute model set, the per-op base_cost above is ignored.
// TODO(cost-model): read these from BackendHandler instead of hardcoding 910B.
constexpr int64_t kFastMemoryCapacity = 1LL << 30;  // legacy single-pool fallback
constexpr int64_t kSlowMemoryBandwidth = 10;        // DDR bandwidth
constexpr int64_t kNativeW = 128;
constexpr int64_t kNativeH = 128;
constexpr int kNumCubeCores = 24;               // AIC cores (matmul)
constexpr int kNumVectorCores = 48;             // AIV cores (pointwise / reduction)
constexpr int64_t kL1Capacity = 512 * 1024;     // per-cube L1/Mat operand pool
constexpr int64_t kCubeCapacity = 128 * 1024;   // per-cube L0c accumulator
constexpr int64_t kVecCapacity = 192 * 1024;    // per-vector UB
constexpr int64_t kCubeComputeCost = 64;        // per 16x16x16 cube fractal (memory-bound default)
constexpr int64_t kVectorComputeCost = 1;       // per vector SIMD step
constexpr int64_t kVectorLanes = 256;           // elements per vector SIMD step
constexpr int64_t kKernelFillCost = 10000;      // per-kernel pipeline fill

// Core counts + on-chip capacities. Defaults are the 910B values above; the
// real ones are read from the configured backend's SoC (so the safe-UB cap and
// 950 specs are picked up automatically). Compute-cost / bandwidth params are
// cost-model calibration (not in the SoC) and stay as tuned constants.
struct HwParams {
  int num_cube_cores = kNumCubeCores;
  int num_vector_cores = kNumVectorCores;
  int64_t l1_capacity = kL1Capacity;      // per-cube Mat
  int64_t cube_capacity = kCubeCapacity;  // per-cube Acc (L0c)
  int64_t vec_capacity = kVecCapacity;    // per-vector Vec (UB)
};

// Read the topology + capacities from the configured backend's SoC (SoC -> Die
// -> Cluster -> Core -> Mem). Falls back to the 910B defaults when no backend is
// configured (e.g. standalone `passes.auto_fuse()` with no PassContext/backend).
HwParams ReadHwParams() {
  HwParams p;
  if (!backend::BackendConfig::IsConfigured()) {
    return p;
  }
  const backend::SoC& soc = backend::GetBackend()->GetSoC();
  int cube = 0, vec = 0;
  int64_t l1 = 0, acc = 0, ub = 0;
  for (const auto& [die, die_n] : soc.GetDieCounts()) {
    for (const auto& [cluster, cl_n] : die.GetClusterCounts()) {
      for (const auto& [core, core_n] : cluster.GetCoreCounts()) {
        const int n = die_n * cl_n * core_n;
        if (core.GetCoreType() == CoreType::CUBE) {
          cube += n;
          for (const auto& m : core.GetMems()) {
            if (m.GetMemType() == MemorySpace::Mat) l1 = static_cast<int64_t>(m.GetMemSize());
            if (m.GetMemType() == MemorySpace::Acc) acc = static_cast<int64_t>(m.GetMemSize());
          }
        } else if (core.GetCoreType() == CoreType::VECTOR) {
          vec += n;
          for (const auto& m : core.GetMems()) {
            if (m.GetMemType() == MemorySpace::Vec) ub = static_cast<int64_t>(m.GetMemSize());
          }
        }
      }
    }
  }
  if (cube > 0) {
    p.num_cube_cores = cube;
    if (l1 > 0) p.l1_capacity = l1;
    if (acc > 0) p.cube_capacity = acc;
  }
  if (vec > 0) {
    p.num_vector_cores = vec;
    if (ub > 0) p.vec_capacity = ub;
  }
  return p;
}

// A call is an allocation (output-buffer creation), not a compute op.
bool IsAllocCall(const CallPtr& call) {
  const std::string& name = call->op_->name_;
  return name.find("create") != std::string::npos || name.find("alloc") != std::string::npos;
}

// A statement is a compute op iff it is `var = <call>` for a non-allocation call.
bool IsComputeOp(const StmtPtr& stmt, CallPtr* out_call) {
  auto assign = As<AssignStmt>(stmt);
  if (assign == nullptr) {
    return false;
  }
  auto call = As<Call>(assign->value_);
  if (call == nullptr || IsAllocCall(call)) {
    return false;
  }
  *out_call = call;
  return true;
}

// Map a PyPTO op/kernel name to a tiling cost category. Broadcast folds into
// Pointwise (its FIXED operand is shape-inferred from the size-1 dim). Memory /
// cross-core / sync ops are not graph nodes, so they never reach this point.
// (For orchestration kernel calls the name is the kernel's — matmul kernels must
// be named *matmul*; body inspection is a TODO.)
::OpType ClassifyOp(const CallPtr& call) {
  const std::string& n = call->op_->name_;
  auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  auto ends = [&](const char* s) {
    const size_t l = std::char_traits<char>::length(s);
    return n.size() >= l && n.compare(n.size() - l, l, s) == 0;
  };
  if (has("matmul") || has("gemv")) {
    return ::OpType::MatMul;
  }
  if (ends(".sum") || ends(".max") || ends(".min") || has("row_sum") ||
      has("row_max") || has("row_min") || has("col_sum") || has("col_max") || has("col_min")) {
    return ::OpType::Reduction;
  }
  if (has("gather") || has("scatter") || has("sort") || has("transpose") || has("reshape") ||
      has("concat") || has("assemble")) {
    return ::OpType::Opaque;  // data-dependent / relayout — un-fusable barrier
  }
  return ::OpType::Pointwise;  // elementwise / unary / cast / expand(broadcast)
}

// Build the MLSys solver `Problem` (op+tensor DAG) from a function, reusing
// `BuildStmtDependencyGraph` for sound op-dependency edges.
class ProblemBuilder {
 public:
  ::Problem problem;
  std::vector<std::string> op_labels;     // per-op kernel/op name, for readable logging
  std::vector<const Stmt*> op_stmts;      // per-op source AssignStmt (op index -> Stmt), for the emit

  void Build(const FunctionPtr& func, const ProgramPtr& prog) {
    problem.fast_memory_capacity = kFastMemoryCapacity;
    problem.slow_memory_bandwidth = kSlowMemoryBandwidth;
    problem.native_w = kNativeW;
    problem.native_h = kNativeH;
    // Topology + on-chip capacities from the configured backend's SoC (the safe
    // UB cap and 950 specs are picked up automatically; 910B defaults if none).
    const HwParams hw = ReadHwParams();
    problem.num_cube_cores = hw.num_cube_cores;
    problem.num_vector_cores = hw.num_vector_cores;
    problem.l1_capacity = hw.l1_capacity;
    problem.cube_capacity = hw.cube_capacity;
    problem.vec_capacity = hw.vec_capacity;
    // Cost-model calibration (not in the SoC).
    problem.cube_compute_cost = kCubeComputeCost;
    problem.vector_compute_cost = kVectorComputeCost;
    problem.vector_lanes = kVectorLanes;
    problem.kernel_fill_cost = kKernelFillCost;
    problem.ddr_atomic_add = true;  // 910B SetAtomicAdd (split-K partials merge in DDR)

    // 1. In-direction params are graph-input tensors (Out/InOut params are
    //    output buffers, not inputs).
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::In) {
        in_params_.insert(func->params_[i].get());
        TensorId(func->params_[i]);
      }
    }

    // 2. Sound op-dependency graph (handles Out/InOut/SSA per RFC #1026).
    stmt_dep::StmtDependencyGraph dep = stmt_dep::BuildStmtDependencyGraph(func->body_, prog);

    // 3. First pass: register each compute op's output tensor (skip allocations).
    std::vector<std::pair<const Stmt*, CallPtr>> ops;
    for (const StmtPtr& stmt : dep.stmts) {
      CallPtr call;
      if (!IsComputeOp(stmt, &call)) {
        continue;
      }
      auto assign = As<AssignStmt>(stmt);
      stmt_output_[stmt.get()] = TensorId(assign->var_);
      ops.emplace_back(stmt.get(), call);
    }

    // 4. Second pass: emit ops. Inputs = predecessor-op outputs (from the
    //    dependency graph) + In-param args. Out-buffers/allocs fall out because
    //    they are never registered as tensors.
    for (const auto& entry : ops) {
      const Stmt* stmt = entry.first;
      const CallPtr& call = entry.second;
      ::Op sop;
      sop.type = ClassifyOp(call);
      std::set<size_t> inputs;
      auto pit = dep.predecessors.find(stmt);
      if (pit != dep.predecessors.end()) {
        for (const Stmt* pred : pit->second) {
          auto oit = stmt_output_.find(pred);
          if (oit != stmt_output_.end()) {
            inputs.insert(oit->second);
          }
        }
      }
      for (const ExprPtr& arg : call->args_) {
        auto var = AsVarLike(arg);
        if (var != nullptr && in_params_.count(var.get()) != 0) {
          inputs.insert(tensor_index_.at(var.get()));
        }
      }
      sop.inputs.assign(inputs.begin(), inputs.end());
      const size_t out = stmt_output_.at(stmt);
      sop.outputs.push_back(out);
      const ::Tensor& ot = problem.tensors[out];
      sop.base_cost = ComputeCost(sop.type, ot.width, ot.height, ot.width);
      problem.ops.push_back(std::move(sop));
      op_labels.push_back(call->op_->name_);
      op_stmts.push_back(stmt);
    }
  }

 private:
  std::unordered_map<const Var*, size_t> tensor_index_;
  std::unordered_map<const Stmt*, size_t> stmt_output_;
  std::unordered_set<const Var*> in_params_;

  size_t TensorId(const VarPtr& var) {
    const Var* raw = var.get();
    auto it = tensor_index_.find(raw);
    if (it != tensor_index_.end()) {
      return it->second;
    }
    auto tt = As<TensorType>(var->GetType());
    CHECK(tt != nullptr) << "AutoFuse: variable '" << var->name_hint_ << "' is not tensor-typed";
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

// Dump the extracted DAG as a competition-format JSON instance (for
// visualization via 3rdparty/mlsys26/scripts/visualize.py). Hand-rolled JSON.
void DumpProblemJson(const ::Problem& p, const std::string& path) {
  std::ofstream f(path);
  if (!f) {
    return;
  }
  const size_t nt = p.tensors.size();
  const size_t no = p.ops.size();
  f << "{\n  \"widths\": [";
  for (size_t i = 0; i < nt; ++i) f << (i ? "," : "") << p.tensors[i].width;
  f << "],\n  \"heights\": [";
  for (size_t i = 0; i < nt; ++i) f << (i ? "," : "") << p.tensors[i].height;
  f << "],\n  \"inputs\": [";
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "[";
    for (size_t j = 0; j < p.ops[i].inputs.size(); ++j) f << (j ? "," : "") << p.ops[i].inputs[j];
    f << "]";
  }
  f << "],\n  \"outputs\": [";
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "[";
    for (size_t j = 0; j < p.ops[i].outputs.size(); ++j) f << (j ? "," : "") << p.ops[i].outputs[j];
    f << "]";
  }
  f << "],\n  \"base_costs\": [";
  for (size_t i = 0; i < no; ++i) f << (i ? "," : "") << p.ops[i].base_cost;
  f << "],\n  \"op_types\": [";
  for (size_t i = 0; i < no; ++i)
    f << (i ? "," : "") << (p.ops[i].type == ::OpType::MatMul ? "\"MatMul\"" : "\"Pointwise\"");
  f << "],\n  \"fast_memory_capacity\": " << p.fast_memory_capacity
    << ",\n  \"slow_memory_bandwidth\": " << p.slow_memory_bandwidth << ",\n  \"native_granularity\": ["
    << p.native_w << ", " << p.native_h << "]\n}\n";
}

// Dump the solver's DECISION (fusion groups + per-group tile/latency/retain) as
// JSON for `3rdparty/mlsys26/scripts/visualize.py solution <dag.json> <sol.json>`.
void DumpSolutionJson(const ::Solution& sol, const std::string& path) {
  std::ofstream f(path);
  if (!f) {
    return;
  }
  const size_t ns = sol.num_steps();
  f << "{\n  \"subgraphs\": [";
  for (size_t s = 0; s < ns; ++s) {
    const std::vector<size_t>& ops = sol.step(s).subgraph.ops();
    f << (s ? "," : "") << "[";
    for (size_t j = 0; j < ops.size(); ++j) f << (j ? "," : "") << ops[j];
    f << "]";
  }
  f << "],\n  \"granularities\": [";  // per-group [w,h,k] — the tiling decision
  for (size_t s = 0; s < ns; ++s) {
    const ::TileConfig& c = sol.step(s).config;
    f << (s ? "," : "") << "[" << c.w << "," << c.h << "," << c.k << "]";
  }
  f << "],\n  \"subgraph_latencies\": [";
  for (size_t s = 0; s < ns; ++s) f << (s ? "," : "") << sol.step_latency(s);
  f << "],\n  \"tensors_to_retain\": [";
  for (size_t s = 0; s < ns; ++s) {
    const std::vector<size_t>& rt = sol.step(s).retain_these.underlying();
    f << (s ? "," : "") << "[";
    for (size_t j = 0; j < rt.size(); ++j) f << (j ? "," : "") << rt[j];
    f << "]";
  }
  f << "]\n}\n";
}

ExprPtr MakeIndex(int64_t v, const Span& span) {
  return std::make_shared<ConstInt>(v, DataType::INDEX, span);
}

ExprPtr MakeIndexTuple(const std::vector<int64_t>& values, const Span& span) {
  std::vector<ExprPtr> elements;
  elements.reserve(values.size());
  for (auto v : values) elements.push_back(MakeIndex(v, span));
  return std::make_shared<MakeTuple>(std::move(elements), span);
}

// Static 2D extent of a tensor-typed expr's type, or {-1,-1} if not a static 2D
// TensorType (dynamic / wrong rank — caller bails out of pipelining).
std::pair<int64_t, int64_t> Static2DShape(const TypePtr& type) {
  auto tt = As<TensorType>(type);
  if (tt == nullptr || tt->shape_.size() != 2) {
    return {-1, -1};
  }
  auto r = As<ConstInt>(tt->shape_[0]);
  auto c = As<ConstInt>(tt->shape_[1]);
  if (r == nullptr || c == nullptr) {
    return {-1, -1};
  }
  return {r->value_, c->value_};
}

// The solver's per-group output tile + contraction tile (TileConfig). `w` tiles
// the output width (N), `h` the output height (M), `k` the contraction (K).
struct SolverTile {
  int64_t w = 0;
  int64_t h = 0;
  int64_t k = 0;
};

ExprPtr MakeTuple2(ExprPtr a, ExprPtr b, const Span& sp) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{std::move(a), std::move(b)}, sp);
}

// Build the compute for ONE [h,w] output tile: `out = a[mi:mi+h, :] @ b[:, ni:ni+w]`,
// streaming the contraction K in `k`-strips with a stage=2 software pipeline (the
// DDR<->L1 / GM->Mat double-buffer that justifies the roofline `max(compute,ddr)`).
// Mirrors AutoTileMatmulL0's K-loop builder, but at the TENSOR level (tensor.slice
// + tensor.matmul/_acc, GM->Mat) since AutoFuse runs before ConvertTensorToTileOps.
// `mi`/`ni` are element offsets along M/N (loop vars, or constant 0). Falls back to
// a single matmul over the full K when K can't be split into >=2 clean strips.
std::vector<StmtPtr> BuildTileMatmul(const ExprPtr& a, const ExprPtr& b, const ExprPtr& mi,
                                     const ExprPtr& ni, int64_t h, int64_t w, int64_t K, int64_t k,
                                     const DataType& dtype, const VarPtr& out_var,
                                     const std::string& base, const Span& sp) {
  auto& reg = OpRegistry::GetInstance();
  const std::vector<std::pair<std::string, std::any>> mm_kw = {
      {"a_trans", false}, {"b_trans", false}, {"c_matrix_nz", false}, {"out_dtype", dtype}};
  const std::vector<std::pair<std::string, std::any>> acc_kw = {{"a_trans", false}, {"b_trans", false}};

  if (k <= 0 || K % k != 0 || K / k < 2) {
    // No k-pipeline (one strip): a single matmul over the full K for this tile.
    auto at = reg.Create("tensor.slice", {a, MakeIndexTuple({h, K}, sp), MakeTuple2(mi, MakeIndex(0, sp), sp)}, sp);
    auto av = std::make_shared<Var>(base + "_a_t", at->GetType(), sp);
    auto bt = reg.Create("tensor.slice", {b, MakeIndexTuple({K, w}, sp), MakeTuple2(MakeIndex(0, sp), ni, sp)}, sp);
    auto bv = std::make_shared<Var>(base + "_b_t", bt->GetType(), sp);
    auto mm = reg.Create("tensor.matmul", {av, bv}, mm_kw, sp);
    return {std::make_shared<AssignStmt>(av, at, sp), std::make_shared<AssignStmt>(bv, bt, sp),
            std::make_shared<AssignStmt>(out_var, mm, sp)};
  }

  // acc accumulates over the K-strips; double-buffered via stage=2.
  auto acc_call = reg.Create("tensor.create", {MakeIndexTuple({h, w}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto acc_var = std::make_shared<Var>(base + "_acc_init", acc_call->GetType(), sp);
  auto acc_assign = std::make_shared<AssignStmt>(acc_var, acc_call, sp);
  auto ko = std::make_shared<Var>(base + "_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
  auto c_iter = std::make_shared<IterArg>(base + "_c", acc_var->GetType(), acc_var, sp);

  // Per-iteration k-strip slices: a[mi:mi+h, ko:ko+k], b[ko:ko+k, ni:ni+w].
  auto a_k_call = reg.Create("tensor.slice", {a, MakeIndexTuple({h, k}, sp), MakeTuple2(mi, ko, sp)}, sp);
  auto a_k = std::make_shared<Var>(base + "_a_k", a_k_call->GetType(), sp);
  auto a_k_assign = std::make_shared<AssignStmt>(a_k, a_k_call, sp);
  auto b_k_call = reg.Create("tensor.slice", {b, MakeIndexTuple({k, w}, sp), MakeTuple2(ko, ni, sp)}, sp);
  auto b_k = std::make_shared<Var>(base + "_b_k", b_k_call->GetType(), sp);
  auto b_k_assign = std::make_shared<AssignStmt>(b_k, b_k_call, sp);

  // if (ko == 0): out = matmul(a_k, b_k)  else  out = matmul_acc(c_iter, a_k, b_k).
  auto then_call = reg.Create("tensor.matmul", {a_k, b_k}, mm_kw, sp);
  auto then_var = std::make_shared<Var>(base + "_mm", then_call->GetType(), sp);
  auto then_assign = std::make_shared<AssignStmt>(then_var, then_call, sp);
  auto then_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{then_var}, sp);
  StmtPtr then_body = SeqStmts::Flatten(std::vector<StmtPtr>{then_assign, then_yield}, sp);

  auto else_call = reg.Create("tensor.matmul_acc", {ExprPtr(c_iter), a_k, b_k}, acc_kw, sp);
  auto else_var = std::make_shared<Var>(base + "_mm_acc", else_call->GetType(), sp);
  auto else_assign = std::make_shared<AssignStmt>(else_var, else_call, sp);
  auto else_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{else_var}, sp);
  StmtPtr else_body = SeqStmts::Flatten(std::vector<StmtPtr>{else_assign, else_yield}, sp);

  auto phi = std::make_shared<Var>(base + "_phi", then_call->GetType(), sp);
  auto cond = MakeEq(ko, MakeIndex(0, sp), sp);
  auto if_stmt = std::make_shared<IfStmt>(cond, then_body, std::optional<StmtPtr>(else_body),
                                          std::vector<VarPtr>{phi}, sp);
  auto body_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{phi}, sp);
  StmtPtr body = SeqStmts::Flatten(std::vector<StmtPtr>{a_k_assign, b_k_assign, if_stmt, body_yield}, sp);

  std::vector<std::pair<std::string, std::any>> loop_attrs = {{kPipelineStagesAttr, /*stages=*/2}};
  auto for_stmt = std::make_shared<ForStmt>(ko, MakeIndex(0, sp), MakeIndex(K, sp), MakeIndex(k, sp),
                                            std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{out_var},
                                            sp, ForKind::Pipeline, /*chunk_config=*/std::nullopt,
                                            std::move(loop_attrs));
  return {acc_assign, for_stmt};
}

// Realize the solver's tile for a `c = tensor.matmul(a, b)`: tile the output
// `[M,N]` into `[h,w]` regions (so each fits L0c) and stream the contraction in
// `k`-strips per tile. Returns the replacement statements, or nullopt if the
// matmul is not eligible (non-default orientation / non-static shapes / tile not
// dividing the output). v0 emits Sequential output-tile loops; cross-core
// Parallel distribution of those tiles is the next increment.
std::optional<std::vector<StmtPtr>> TileMatmul(const AssignStmtPtr& assign, SolverTile tile,
                                              const std::string& name) {
  auto call = As<Call>(assign->value_);
  if (call == nullptr || call->op_ == nullptr || call->op_->name_ != "tensor.matmul" ||
      call->args_.size() != 2) {
    return std::nullopt;
  }
  const ExprPtr a = call->args_[0];
  const ExprPtr b = call->args_[1];
  const VarPtr c_var = assign->var_;
  auto ct = As<TensorType>(c_var->GetType());
  if (ct == nullptr) {
    return std::nullopt;
  }
  const DataType dtype = ct->dtype_;
  const auto [M, N] = Static2DShape(c_var->GetType());
  const auto [lM, lK] = Static2DShape(a->GetType());
  const auto [rK, rN] = Static2DShape(b->GetType());
  // Default-orientation guard: a[M,K] @ b[K,N] -> c[M,N] (any -1 fails a compare).
  const int64_t K = lK;
  if (M < 0 || lM != M || rN != N || rK != K) {
    return std::nullopt;
  }

  // Clamp the output tile to the output and require clean division.
  int64_t h = (tile.h > 0 && tile.h < M) ? tile.h : M;
  int64_t w = (tile.w > 0 && tile.w < N) ? tile.w : N;
  if (M % h != 0 || N % w != 0) {
    return std::nullopt;
  }
  const int64_t num_m = M / h;
  const int64_t num_n = N / w;

  const Span sp = assign->span_;
  const std::string base = c_var->name_hint_;

  // The solver's tile is the whole output: no output loop — just the k-pipeline
  // (writing directly into the original output var), wrapped in one InCore kernel.
  if (num_m == 1 && num_n == 1) {
    auto stmts = BuildTileMatmul(a, b, MakeIndex(0, sp), MakeIndex(0, sp), M, N, K, tile.k, dtype, c_var, base, sp);
    return std::vector<StmtPtr>{
        std::make_shared<InCoreScopeStmt>(std::nullopt, name, SeqStmts::Flatten(std::move(stmts), sp), sp)};
  }

  // Output spatial tiling distributed ACROSS CORES via the standard chunk path:
  // an AutoInCore scope wraps nested chunked PARALLEL loops over the [w,h] output
  // tiles (tile-index loops; element offset = idx*tile). Each iteration computes
  // the [h,w] tile (k-pipelined) and assembles it into the output. The existing
  // SplitChunkedLoops -> InterchangeChunkLoops -> OutlineIncoreScopes passes then
  // distribute the tiles across cores and outline the per-tile kernel. The output
  // tensor.create stays OUTSIDE the scope, so the full [M,N] output is a DDR
  // tensor and only the [h,w] tile is on-chip.
  auto& reg = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);

  auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
  auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);

  // Loop vars are TILE INDICES (0..num_m / 0..num_n); element offset = idx * tile.
  auto mt = std::make_shared<Var>(base + "_mt", index_type, sp);
  auto nt = std::make_shared<Var>(base + "_nt", index_type, sp);
  auto mi = MakeMul(mt, MakeIndex(h, sp), sp);
  auto ni = MakeMul(nt, MakeIndex(w, sp), sp);
  auto c_m = std::make_shared<IterArg>(base + "_cm", c_init->GetType(), c_init, sp);
  auto c_n = std::make_shared<IterArg>(base + "_cn", c_init->GetType(), c_m, sp);

  // Inner N-loop body: compute the [h,w] tile (k-pipeline) and assemble into c.
  auto tile_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
  auto tile_var = std::make_shared<Var>(base + "_tile", tile_type, sp);
  std::vector<StmtPtr> n_body_stmts = BuildTileMatmul(a, b, mi, ni, h, w, K, tile.k, dtype, tile_var, base, sp);
  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_n), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  auto c_new = std::make_shared<Var>(base + "_cnew", asm_call->GetType(), sp);
  n_body_stmts.push_back(std::make_shared<AssignStmt>(c_new, asm_call, sp));
  n_body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_new}, sp));
  StmtPtr n_body = SeqStmts::Flatten(std::move(n_body_stmts), sp);

  auto c_row = std::make_shared<Var>(base + "_crow", c_init->GetType(), sp);
  ChunkConfig n_chunk{MakeIndex(num_n, sp), ChunkPolicy::LeadingFull};
  auto n_loop = std::make_shared<ForStmt>(nt, MakeIndex(0, sp), MakeIndex(num_n, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_n}, n_body, std::vector<VarPtr>{c_row}, sp,
                                          ForKind::Parallel, n_chunk);

  StmtPtr m_body = SeqStmts::Flatten(
      std::vector<StmtPtr>{n_loop, std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_row}, sp)}, sp);
  ChunkConfig m_chunk{MakeIndex(num_m, sp), ChunkPolicy::LeadingFull};
  auto m_loop = std::make_shared<ForStmt>(mt, MakeIndex(0, sp), MakeIndex(num_m, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_m}, m_body, std::vector<VarPtr>{c_var}, sp,
                                          ForKind::Parallel, m_chunk);

  auto scope = std::make_shared<AutoInCoreScopeStmt>(std::nullopt, name, m_loop, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// Realize the solver's [w,h] tiling for a single POINTWISE op (`c = pw(a, ...)`):
// tile the output into [w,h] regions distributed across the VECTOR cores via the
// standard AutoInCore chunked-parallel path — each tile a per-tile kernel that
// slices its output-shaped operands, applies the op (preserving its kwargs via
// MutableCopy), and assembles into the DDR output. Same wrapper as TileMatmul's
// spatial path, but the per-tile body is the op on [h,w] slices (no k-loop).
// nullopt if not eligible. TODO: share the chunked-parallel wrapper with TileMatmul.
std::optional<std::vector<StmtPtr>> TilePointwise(const AssignStmtPtr& assign, SolverTile tile,
                                                  const std::string& name) {
  auto call = As<Call>(assign->value_);
  if (call == nullptr || call->op_ == nullptr || ClassifyOp(call) != ::OpType::Pointwise) {
    return std::nullopt;
  }
  const VarPtr c_var = assign->var_;
  auto ct = As<TensorType>(c_var->GetType());
  if (ct == nullptr) {
    return std::nullopt;
  }
  const DataType dtype = ct->dtype_;
  const auto [M, N] = Static2DShape(c_var->GetType());
  if (M < 0) {
    return std::nullopt;
  }
  int64_t h = (tile.h > 0 && tile.h < M) ? tile.h : M;
  int64_t w = (tile.w > 0 && tile.w < N) ? tile.w : N;
  if (M % h != 0 || N % w != 0) {
    return std::nullopt;
  }
  const int64_t num_m = M / h, num_n = N / w;
  if (num_m == 1 && num_n == 1) {
    return std::nullopt;  // whole output is one tile -> the plain InCore scope handles it
  }

  const Span sp = assign->span_;
  const std::string base = c_var->name_hint_;
  auto& reg = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);

  auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
  auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);
  auto mt = std::make_shared<Var>(base + "_mt", index_type, sp);
  auto nt = std::make_shared<Var>(base + "_nt", index_type, sp);
  auto mi = MakeMul(mt, MakeIndex(h, sp), sp);
  auto ni = MakeMul(nt, MakeIndex(w, sp), sp);
  auto c_m = std::make_shared<IterArg>(base + "_cm", c_init->GetType(), c_init, sp);
  auto c_n = std::make_shared<IterArg>(base + "_cn", c_init->GetType(), c_m, sp);

  // Per-tile body: slice each output-shaped operand to [h,w]@[mi,ni]; apply the op.
  std::vector<ExprPtr> tile_args;
  std::vector<StmtPtr> body_stmts;
  for (const ExprPtr& arg : call->args_) {
    const auto [aM, aN] = Static2DShape(arg->GetType());
    if (aM == M && aN == N) {
      auto sl = reg.Create("tensor.slice", {arg, MakeIndexTuple({h, w}, sp), MakeTuple2(mi, ni, sp)}, sp);
      auto sv = std::make_shared<Var>(base + "_in", sl->GetType(), sp);
      body_stmts.push_back(std::make_shared<AssignStmt>(sv, sl, sp));
      tile_args.push_back(sv);
    } else {
      tile_args.push_back(arg);  // scalar / non-output-shaped operand kept as-is
    }
  }
  // Re-create the op (not MutableCopy) so its result type is re-inferred from the
  // [h,w] tile-shaped args — a copied call keeps the original full-output type and
  // fails the print->parse roundtrip. The original kwargs (e.g. the scalar operand
  // for `tensor.adds`) are preserved.
  auto pw = reg.Create(call->op_->name_, tile_args, call->kwargs_, sp);
  auto tile_var = std::make_shared<Var>(base + "_tile", pw->GetType(), sp);
  body_stmts.push_back(std::make_shared<AssignStmt>(tile_var, pw, sp));

  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_n), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  auto c_new = std::make_shared<Var>(base + "_cnew", asm_call->GetType(), sp);
  body_stmts.push_back(std::make_shared<AssignStmt>(c_new, asm_call, sp));
  body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_new}, sp));
  StmtPtr n_body = SeqStmts::Flatten(std::move(body_stmts), sp);

  auto c_row = std::make_shared<Var>(base + "_crow", c_init->GetType(), sp);
  ChunkConfig n_chunk{MakeIndex(num_n, sp), ChunkPolicy::LeadingFull};
  auto n_loop = std::make_shared<ForStmt>(nt, MakeIndex(0, sp), MakeIndex(num_n, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_n}, n_body, std::vector<VarPtr>{c_row}, sp,
                                          ForKind::Parallel, n_chunk);
  StmtPtr m_body = SeqStmts::Flatten(
      std::vector<StmtPtr>{n_loop, std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_row}, sp)}, sp);
  ChunkConfig m_chunk{MakeIndex(num_m, sp), ChunkPolicy::LeadingFull};
  auto m_loop = std::make_shared<ForStmt>(mt, MakeIndex(0, sp), MakeIndex(num_m, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_m}, m_body, std::vector<VarPtr>{c_var}, sp,
                                          ForKind::Parallel, m_chunk);
  auto scope = std::make_shared<AutoInCoreScopeStmt>(std::nullopt, name, m_loop, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// Realize the solver's FUSED chained matmul: two matmuls `T = A @ B` and
// `C = T @ D` placed in ONE group, where MM2 consumes MM1's output T. The
// intermediate T never touches DDR (the fusion) — it is recomputed on-chip per
// output tile. Tile C's output `[M,N]` into `[h,w]` regions across cores (the
// parallel outer, same wrapper as TileMatmul); each tile's body is the inner
// serial chain: T_band = `A[mi:mi+h, :] @ B` ([h,K2], on-chip), then
// `C_tile = T_band @ D[:, ni:ni+w]` ([h,w]). MM1 keeps the DDR<->L1 k-pipeline
// (its operands stream from DDR); MM2 is a single matmul (its left operand T_band
// is already on-chip). Returns nullopt if the pair is not a default-orientation
// static-shape chain or the tile does not divide the output. Constraint: the
// per-tile T_band (MM1's output) must fit L0c — larger intermediates need the
// AutoTileMatmulL0 M/N-tiling work. TODO: share the wrapper with TileMatmul.
std::optional<std::vector<StmtPtr>> TileChainedMatmul(const AssignStmtPtr& mm1,
                                                      const AssignStmtPtr& mm2, SolverTile tile,
                                                      const std::string& name) {
  auto c1 = As<Call>(mm1->value_);  // T = matmul(A, B)
  auto c2 = As<Call>(mm2->value_);  // C = matmul(T, D)
  if (c1 == nullptr || c2 == nullptr || c1->op_ == nullptr || c2->op_ == nullptr) {
    return std::nullopt;
  }
  if (c1->op_->name_ != "tensor.matmul" || c2->op_->name_ != "tensor.matmul" ||
      c1->args_.size() != 2 || c2->args_.size() != 2) {
    return std::nullopt;
  }
  const ExprPtr A = c1->args_[0];
  const ExprPtr B = c1->args_[1];
  const ExprPtr D = c2->args_[1];  // c2->args_[0] is T (== mm1 output), verified by the caller
  const VarPtr c_var = mm2->var_;
  auto ct = As<TensorType>(c_var->GetType());
  if (ct == nullptr) {
    return std::nullopt;
  }
  const DataType dtype = ct->dtype_;
  const auto [M, N] = Static2DShape(c_var->GetType());
  const auto [aM, aK1] = Static2DShape(A->GetType());
  const auto [bK1, bK2] = Static2DShape(B->GetType());
  const auto [dK2, dN] = Static2DShape(D->GetType());
  // Default-orientation chain: a[M,K1]@b[K1,K2] -> T[M,K2]; T@d[K2,N] -> c[M,N].
  if (M < 0 || aM != M || bK1 != aK1 || dN != N || dK2 != bK2) {
    return std::nullopt;
  }
  const int64_t K1 = aK1;
  const int64_t K2 = bK2;

  int64_t h = (tile.h > 0 && tile.h < M) ? tile.h : M;
  int64_t w = (tile.w > 0 && tile.w < N) ? tile.w : N;
  if (M % h != 0 || N % w != 0) {
    return std::nullopt;
  }
  const int64_t num_m = M / h;
  const int64_t num_n = N / w;

  const Span sp = mm2->span_;
  const std::string base = c_var->name_hint_;
  auto& reg = OpRegistry::GetInstance();

  // Inner serial chain for one output tile at element offset [mi,ni]:
  //   T_band = A[mi:mi+h, :] @ B            -> [h,K2]  (k-pipelined: A streams from DDR)
  //   out_tile = T_band @ D[:, ni:ni+w]     -> [h,w]   (single: T_band is on-chip)
  auto build_chain = [&](const ExprPtr& mi, const ExprPtr& ni, const VarPtr& out_tile) {
    std::vector<StmtPtr> stmts;
    auto tband_type =
        std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(K2, sp)}, dtype);
    auto tband = std::make_shared<Var>(base + "_tband", tband_type, sp);
    auto s1 = BuildTileMatmul(A, B, mi, MakeIndex(0, sp), h, K2, K1, tile.k, dtype, tband, base + "_t", sp);
    auto s2 = BuildTileMatmul(tband, D, MakeIndex(0, sp), ni, h, w, K2, /*k=*/0, dtype, out_tile, base + "_c", sp);
    for (auto& s : s1) stmts.push_back(std::move(s));
    for (auto& s : s2) stmts.push_back(std::move(s));
    return stmts;
  };

  // Whole output is one tile: the fused chain in one InCore kernel (T on-chip).
  if (num_m == 1 && num_n == 1) {
    auto stmts = build_chain(MakeIndex(0, sp), MakeIndex(0, sp), c_var);
    return std::vector<StmtPtr>{
        std::make_shared<InCoreScopeStmt>(std::nullopt, name, SeqStmts::Flatten(std::move(stmts), sp), sp)};
  }

  // Spatial output tiling distributed across cores (same chunked-parallel wrapper
  // as TileMatmul): the [w,h] output tiles fan out across cores, each per-tile
  // kernel runs the inner serial chain with its T_band on-chip.
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
  auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
  auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);
  auto mt = std::make_shared<Var>(base + "_mt", index_type, sp);
  auto nt = std::make_shared<Var>(base + "_nt", index_type, sp);
  auto mi = MakeMul(mt, MakeIndex(h, sp), sp);
  auto ni = MakeMul(nt, MakeIndex(w, sp), sp);
  auto c_m = std::make_shared<IterArg>(base + "_cm", c_init->GetType(), c_init, sp);
  auto c_n = std::make_shared<IterArg>(base + "_cn", c_init->GetType(), c_m, sp);

  auto tile_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
  auto tile_var = std::make_shared<Var>(base + "_tile", tile_type, sp);
  std::vector<StmtPtr> n_body_stmts = build_chain(mi, ni, tile_var);
  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_n), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  auto c_new = std::make_shared<Var>(base + "_cnew", asm_call->GetType(), sp);
  n_body_stmts.push_back(std::make_shared<AssignStmt>(c_new, asm_call, sp));
  n_body_stmts.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_new}, sp));
  StmtPtr n_body = SeqStmts::Flatten(std::move(n_body_stmts), sp);

  auto c_row = std::make_shared<Var>(base + "_crow", c_init->GetType(), sp);
  ChunkConfig n_chunk{MakeIndex(num_n, sp), ChunkPolicy::LeadingFull};
  auto n_loop = std::make_shared<ForStmt>(nt, MakeIndex(0, sp), MakeIndex(num_n, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_n}, n_body, std::vector<VarPtr>{c_row}, sp,
                                          ForKind::Parallel, n_chunk);
  StmtPtr m_body = SeqStmts::Flatten(
      std::vector<StmtPtr>{n_loop, std::make_shared<YieldStmt>(std::vector<ExprPtr>{c_row}, sp)}, sp);
  ChunkConfig m_chunk{MakeIndex(num_m, sp), ChunkPolicy::LeadingFull};
  auto m_loop = std::make_shared<ForStmt>(mt, MakeIndex(0, sp), MakeIndex(num_m, sp), MakeIndex(1, sp),
                                          std::vector<IterArgPtr>{c_m}, m_body, std::vector<VarPtr>{c_var}, sp,
                                          ForKind::Parallel, m_chunk);
  auto scope = std::make_shared<AutoInCoreScopeStmt>(std::nullopt, name, m_loop, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// Rewrite a function body to realize the solver's decision. A matmul becomes its
// own self-scoped tiled kernel (the solver's `[w,h]` output tiling, an InCore
// kernel per tile, the per-tile k-pipeline inside) emitted at the orchestration
// level; two chained matmuls in one group become a single fused kernel (the
// intermediate stays on-chip, see TileChainedMatmul); every other fused group is
// a maximal *contiguous* run of same-group compute stmts wrapped in one
// InCoreScopeStmt. The body is already in SSA dependency order.
StmtPtr EmitFusedScopes(const StmtPtr& body,
                        const std::unordered_map<const Stmt*, size_t>& stmt_group,
                        const std::unordered_map<const Stmt*, SolverTile>& stmt_tile) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = As<SeqStmts>(body)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(body);
  }

  // Detect 2-matmul chains within a group: MM2 = matmul(T, D) where its left
  // operand T is the output of MM1 = matmul(A, B) in the SAME group. Such a pair
  // is emitted as one fused kernel (T on-chip) instead of two separate matmul
  // kernels (T round-tripping DDR). The body is in SSA order, so MM1 precedes MM2.
  auto matmul_call = [](const StmtPtr& s) -> CallPtr {
    auto a = As<AssignStmt>(s);
    if (a == nullptr) return nullptr;
    auto c = As<Call>(a->value_);
    if (c == nullptr || c->op_ == nullptr || c->op_->name_ != "tensor.matmul" || c->args_.size() != 2) {
      return nullptr;
    }
    return c;
  };
  std::unordered_map<const Var*, const Stmt*> mm_out;  // grouped matmul output var -> its stmt
  for (const StmtPtr& stmt : body_stmts) {
    if (stmt_group.find(stmt.get()) == stmt_group.end()) continue;
    if (matmul_call(stmt) != nullptr) {
      mm_out[As<AssignStmt>(stmt)->var_.get()] = stmt.get();
    }
  }
  std::unordered_map<const Stmt*, AssignStmtPtr> chain_head;  // MM1 stmt -> MM2 assign
  for (const StmtPtr& stmt : body_stmts) {
    auto git = stmt_group.find(stmt.get());
    if (git == stmt_group.end()) continue;
    CallPtr c = matmul_call(stmt);
    if (c == nullptr) continue;
    auto lhs = AsVarLike(c->args_[0]);  // T = left operand of MM2
    if (lhs == nullptr) continue;
    auto it = mm_out.find(lhs.get());
    if (it == mm_out.end() || it->second == stmt.get()) continue;  // left operand not a prior matmul
    if (stmt_group.at(it->second) != git->second) continue;        // must be the same fused group
    chain_head[it->second] = As<AssignStmt>(stmt);
  }
  std::unordered_set<const Stmt*> chain_done;  // chain tails already emitted with their head
  std::vector<StmtPtr> top;
  std::vector<StmtPtr> run;
  long run_group = -1;
  // Flush the accumulated run. A lone pointwise op gets the solver's [w,h]
  // cross-core tiling (TilePointwise); everything else is wrapped in one InCore
  // scope (multi-op groups, reductions, or pointwise that needs no tiling).
  auto flush = [&]() {
    if (run.empty()) {
      return;
    }
    const Span scope_span = run.front()->span_;
    const std::string nm = "fused_" + std::to_string(run_group);
    if (run.size() == 1) {
      if (auto assign = As<AssignStmt>(run[0])) {
        auto tit = stmt_tile.find(run[0].get());
        if (tit != stmt_tile.end()) {
          if (auto tiled = TilePointwise(assign, tit->second, nm)) {
            for (auto& s : *tiled) {
              top.push_back(std::move(s));
            }
            run.clear();
            run_group = -1;
            return;
          }
        }
      }
    }
    top.push_back(
        std::make_shared<InCoreScopeStmt>(std::nullopt, nm, SeqStmts::Flatten(run, scope_span), scope_span));
    run.clear();
    run_group = -1;
  };
  for (const StmtPtr& stmt : body_stmts) {
    if (chain_done.count(stmt.get()) != 0) {
      continue;  // chain tail (MM2) already emitted with its head (MM1)
    }
    auto git = stmt_group.find(stmt.get());
    if (git == stmt_group.end()) {  // allocation / return / other non-grouped stmt
      flush();
      top.push_back(stmt);
      continue;
    }
    const long g = static_cast<long>(git->second);
    // A chained matmul pair (MM1 -> MM2 in the same group) is realized as one
    // fused kernel — the parallel-outer output tiling with the inner serial chain
    // and the intermediate on-chip (see TileChainedMatmul).
    auto hit = chain_head.find(stmt.get());
    if (hit != chain_head.end()) {
      auto tit = stmt_tile.find(stmt.get());
      const SolverTile tile = (tit != stmt_tile.end()) ? tit->second : SolverTile{};
      if (auto chained = TileChainedMatmul(As<AssignStmt>(stmt), hit->second, tile, "fused_" + std::to_string(g))) {
        flush();
        for (auto& s : *chained) {
          top.push_back(std::move(s));
        }
        chain_done.insert(hit->second.get());  // MM2 emitted as part of this chain
        continue;
      }
      // Not a tileable chain: fall through — MM1 and MM2 each become standalone
      // matmul kernels (MM2 is not in chain_done, so it is handled when reached).
    }
    // A standalone matmul is realized as its own self-scoped tiled kernel at
    // orchestration level (output `[w,h]` loop + per-tile InCore scope + k-pipeline).
    std::optional<std::vector<StmtPtr>> tiled;
    if (auto assign = As<AssignStmt>(stmt)) {
      auto tit = stmt_tile.find(stmt.get());
      if (tit != stmt_tile.end()) {
        tiled = TileMatmul(assign, tit->second, "fused_" + std::to_string(g));
      }
    }
    if (tiled) {
      flush();
      for (auto& s : *tiled) {
        top.push_back(std::move(s));
      }
    } else {  // other compute op: accumulate into the scoped run
      if (run_group != -1 && run_group != g) {
        flush();
      }
      run_group = g;
      run.push_back(stmt);
    }
  }
  flush();
  return SeqStmts::Flatten(std::move(top), body->span_);
}

ProgramPtr AutoFuseTransform(const ProgramPtr& prog) {
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  bool any_change = false;
  for (const auto& [gvar, func] : prog->functions_) {
    // v0 gate: only functions explicitly marked for auto-fusion. attrs_ is an
    // ordered vector of (key, value) pairs, not a map.
    const bool marked = std::any_of(func->attrs_.begin(), func->attrs_.end(),
                                    [](const auto& kv) { return kv.first == "auto_fuse"; });
    if (!marked) {
      new_functions.emplace(gvar, func);
      continue;
    }
    ProblemBuilder builder;
    builder.Build(func, prog);
    if (builder.problem.ops.empty()) {
      new_functions.emplace(gvar, func);
      continue;
    }
    // Print the intercepted tensor graph (the raw op+tensor DAG the pass sees).
    const ::Problem& p = builder.problem;
    LOG_INFO << "AutoFuse[" << func->name_ << "]: backend "
             << (backend::BackendConfig::IsConfigured() ? "SoC" : "default(910B)") << " — cube_cores="
             << p.num_cube_cores << " vector_cores=" << p.num_vector_cores << " L1=" << p.l1_capacity
             << " Acc=" << p.cube_capacity << " UB=" << p.vec_capacity;
    LOG_INFO << "AutoFuse[" << func->name_ << "]: intercepted tensor graph — " << p.ops.size()
             << " ops, " << p.tensors.size() << " tensors";
    for (size_t i = 0; i < p.ops.size(); ++i) {
      const ::Op& op = p.ops[i];
      std::string ins;
      for (size_t j = 0; j < op.inputs.size(); ++j) {
        ins += (j ? "," : "");
        ins += "t" + std::to_string(op.inputs[j]);
      }
      const ::Tensor& ot = p.tensors[op.outputs[0]];
      LOG_INFO << "  op[" << i << "] " << (op.type == ::OpType::MatMul ? "MatMul   " : "Pointwise")
               << " " << builder.op_labels[i] << "  in={" << ins << "} -> t" << op.outputs[0] << " ["
               << ot.width << "x" << ot.height << "]  cost=" << op.base_cost;
    }

    // Solve, then print the fusion decision: each group's member ops + chosen tile.
    ::DAG dag = ::DAG::build(builder.problem);
    ::Solution sol = ::solve(builder.problem, dag);
    LOG_INFO << "AutoFuse[" << func->name_ << "]: solver -> " << sol.num_steps()
             << " fused group(s), total latency " << sol.total_latency();
    for (size_t s = 0; s < sol.num_steps(); ++s) {
      const ::ScheduleStep& step = sol.step(s);
      std::string members;
      const std::vector<size_t>& gops = step.subgraph.ops();
      for (size_t j = 0; j < gops.size(); ++j) {
        members += (j ? "," : "");
        members += builder.op_labels[gops[j]];
      }
      LOG_INFO << "  group[" << s << "] ops={" << members << "}  tile=" << step.config.w << "x"
               << step.config.h << (step.config.k ? ("x" + std::to_string(step.config.k)) : std::string())
               << "  latency=" << sol.step_latency(s);
    }

    if (const char* dump_dir = std::getenv("PYPTO_AUTOFUSE_DUMP")) {
      const std::string base = std::string(dump_dir) + "/" + func->name_;
      DumpProblemJson(builder.problem, base + ".dag.json");
      DumpSolutionJson(sol, base + ".sol.json");
    }

    // Emit: map each solver op back to its source statement, wrap each fused
    // group in an InCoreScopeStmt, and realize the chosen tile (step.config) for
    // matmuls — the output `[w,h]` tiling + the per-tile k-pipeline.
    std::unordered_map<const Stmt*, size_t> stmt_group;
    std::unordered_map<const Stmt*, SolverTile> stmt_tile;  // group's [w,h,k] tile, for matmul tiling
    for (size_t s = 0; s < sol.num_steps(); ++s) {
      const ::TileConfig& cfg = sol.step(s).config;
      const SolverTile tile{cfg.w, cfg.h, cfg.k};
      for (size_t op_idx : sol.step(s).subgraph.ops()) {
        const Stmt* stmt = builder.op_stmts[op_idx];
        stmt_group[stmt] = s;
        stmt_tile[stmt] = tile;
      }
    }
    auto new_func = MutableCopy(func);
    new_func->body_ = EmitFusedScopes(func->body_, stmt_group, stmt_tile);
    // Drop the marker once fused: the body is now an InCore-scoped kernel graph,
    // not a flat tensor-op DAG, so the pass is idempotent (a second run no-ops).
    new_func->attrs_.erase(
        std::remove_if(new_func->attrs_.begin(), new_func->attrs_.end(),
                       [](const auto& kv) { return kv.first == "auto_fuse"; }),
        new_func->attrs_.end());
    new_functions.emplace(gvar, new_func);
    any_change = true;
  }
  if (!any_change) {
    return prog;
  }
  auto new_prog = MutableCopy(prog);
  new_prog->functions_ = std::move(new_functions);
  return new_prog;
}

}  // namespace

Pass AutoFuse() { return CreateProgramPass(AutoFuseTransform, "AutoFuse", {}); }

}  // namespace pass
}  // namespace ir
}  // namespace pypto
