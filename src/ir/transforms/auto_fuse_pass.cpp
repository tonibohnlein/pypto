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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/transforms/pass_context.h"
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
#include "pypto/ir/transforms/base/visitor.h"
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

// Hardware parameters. v0 hardcodes the Ascend 910B machine model (mirrors
// `set_910b` in 3rdparty/mlsys26/test/ascend_910b_test.cpp); the solver derives
// per-op compute from tile geometry (the grounded pto-isa fractal/vector model).
// TODO(cost-model): read these from BackendHandler instead of hardcoding 910B.
constexpr int64_t kFastMemoryCapacity = 1LL << 30;  // single-pool capacity hint
constexpr int kNumCubeCores = 24;               // AIC cores (matmul)
constexpr int kNumVectorCores = 48;             // AIV cores (pointwise / reduction)
constexpr int64_t kL1Capacity = 512 * 1024;     // per-cube L1/Mat operand pool
constexpr int64_t kCubeCapacity = 128 * 1024;   // per-cube L0c accumulator
constexpr int64_t kVecCapacity = 192 * 1024;    // per-vector UB
constexpr int64_t kCubeComputeCost = 1;         // grounded per-repeat multiplier (cyc applies fp32 2x)
constexpr int64_t kKernelFillCost = 10000;      // per-kernel pipeline fill (cycles)

// Grounded pto-isa machine model (Ascend 910B / A2A3). Costs are in CORE CYCLES;
// bandwidths are GiB/s per direction (pto-isa arch_config.hpp). See the solver's
// types.h Problem::cube_freq_hz block for the exact formulas.
constexpr double kCubeFreqHz = 1.85e9;          // core clock (A2A3)
constexpr double kBwGmL1   = 135.0;             // GM->L1 operand reload
constexpr double kBwL0cGm  = 70.0;              // L0C->GM output store
constexpr double kBwL1L0a  = 441.0;             // L1->L0A lhs extract
constexpr double kBwL1L0b  = 220.5;             // L1->L0B rhs extract
constexpr double kBwGmUb   = 100.9;             // GM->UB vector load
constexpr double kBwUbGm   = 188.46;            // UB->GM vector store
// Aggregate HBM bandwidth (GiB/s) shared by all cores — caps the sum of the
// per-core GM pipes (DDR divides across cores up to here). Realistic A3 aggregate
// (~900 GiB/s): par() binds at ~900/135 = 6.7 cores, so reload-bound matmuls
// saturate HBM rather than scale linearly to 24 cores. Perf-sim VALIDATED in the
// saturation regime (pto-isa gml1_multicore: per-core bw = min(135, 900/B) to
// <=0.4%). The exact aggregate is DEVICE-EVAL-PENDING; 900 is the perf-sim estimate.
constexpr double kHbmAggregateGiBps = 900.0;
constexpr int64_t kL0TileM = 128;               // L0 GEMM base M (pto-isa oracle)
constexpr int64_t kL0TileN = 256;               // L0 GEMM base N (pto-isa oracle)
// Grounded vector (AIV) cost: per op = head + slope*repeat + tail cycles, repeat
// = ceil(elems / (vec_reg_bytes/dtype_bytes)). pto-isa A2A3
// cce_costmodel_vector_compute.hpp: 256-byte vreg; vadd head 14 / slope 1 / tail
// 18, vmul slope 2; vreducev2 slope 14.
constexpr int64_t kVecRegBytes = 256;           // vector register size (bytes)
constexpr double kVecOpHead = 14.0;             // per-op pipeline startup
constexpr double kVecOpTail = 18.0;             // per-op drain
constexpr double kVecSlopePw = 2.0;             // elementwise cycles/repeat (vmul-ish)
constexpr double kVecSlopeReduce = 14.0;        // reduction cycles/repeat (vreducev2)

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

// Map a PyPTO DataType to the solver's byte-aware DType. The grounded cube cost
// is dtype-sensitive (fp32 is 4x fp16: kF halves AND cyc_per_repeat doubles), so
// the precision must survive into the cost model. Unmapped types fall back to
// FP32 (the conservative / heaviest cube cost).
::DType MapSolverDType(const DataType& dt) {
  if (dt == DataType::FP16) return ::DType::FP16;
  if (dt == DataType::BF16) return ::DType::BF16;
  if (dt == DataType::INT32) return ::DType::INT32;
  if (dt == DataType::INT16) return ::DType::INT16;
  if (dt == DataType::INT8) return ::DType::INT8;
  if (dt == DataType::BOOL) return ::DType::BOOL;
  return ::DType::FP32;  // FP32 + anything wider/unmapped
}

// Build the MLSys solver `Problem` (op+tensor DAG) from a function, reusing
// `BuildStmtDependencyGraph` for sound op-dependency edges.
class ProblemBuilder {
 public:
  ::Problem problem;
  std::vector<std::string> op_labels;     // per-op kernel/op name, for readable logging
  std::vector<const Stmt*> op_stmts;      // per-op source AssignStmt (op index -> Stmt), for the emit

  // The function is OUT OF SCOPE for AutoFuse v0 (a non-tensor compute output or a
  // dynamic/symbolic tensor shape) — the caller must leave it for legacy lowering. This
  // is NOT a user error (both are legal signatures), so it is a graceful decline, not a
  // CHECK: an auto_fuse-marked function with a symbolic dim must still compile.
  bool declined() const { return declined_; }

  void Build(const FunctionPtr& func, const ProgramPtr& prog) {
    problem.fast_memory_capacity = kFastMemoryCapacity;
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
    problem.kernel_fill_cost = kKernelFillCost;
    // Grounded pto-isa machine model (cycles + per-direction GiB/s bandwidths +
    // hierarchical L1<->L0 cube work). Activates the grounded cost path.
    problem.cube_freq_hz = kCubeFreqHz;
    problem.bw_gm_l1 = kBwGmL1;
    problem.bw_l0c_gm = kBwL0cGm;
    problem.bw_l1_l0a = kBwL1L0a;
    problem.bw_l1_l0b = kBwL1L0b;
    problem.bw_gm_ub = kBwGmUb;
    problem.bw_ub_gm = kBwUbGm;
    problem.hbm_aggregate_gibps = kHbmAggregateGiBps;
    problem.l0_tile_m = kL0TileM;
    problem.l0_tile_n = kL0TileN;
    problem.vec_reg_bytes = kVecRegBytes;
    problem.vec_op_head = kVecOpHead;
    problem.vec_op_tail = kVecOpTail;
    problem.vec_slope_pw = kVecSlopePw;
    problem.vec_slope_reduce = kVecSlopeReduce;

    // 1. In-direction params are graph-input tensors (Out/InOut params are
    //    output buffers, not inputs).
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i < func->param_directions_.size() && func->param_directions_[i] == ParamDirection::In) {
        in_params_.insert(func->params_[i].get());
        // Only tensor-typed In-params are tiled tensors in the solver's model; a scalar
        // In-param (e.g. a broadcast scale) is an operand carried through the emit as-is,
        // never a tracked tensor -> skip it. Registering it would trip TensorId's
        // tensor-type decline and needlessly abandon a fusable function.
        if (As<TensorType>(func->params_[i]->GetType()) != nullptr) TensorId(func->params_[i]);
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
      // Inputs in OPERAND ORDER (call->args_). The solver derives M/N/K
      // positionally (inputs[0]=LHS [K,M], inputs[1]=RHS [N,K]), so the order is
      // load-bearing. Both in-params and predecessor-op outputs are registered
      // in tensor_index_ (steps 1 and 3 above), so one ordered pass over args
      // covers both sources. A std::set would re-sort by tensor index
      // (in-params, registered first, before intermediates), silently swapping a
      // chained matmul's operands (e.g. (A@B)@D's sink [T,D] -> [D,T]).
      std::vector<size_t> inputs;
      std::unordered_set<size_t> seen;
      for (const ExprPtr& arg : call->args_) {
        auto var = AsVarLike(arg);
        if (var == nullptr) {
          continue;
        }
        auto it = tensor_index_.find(var.get());
        if (it == tensor_index_.end()) {
          continue;  // alloc / scalar / Out buffer — not a tracked input tensor
        }
        if (seen.insert(it->second).second) {
          inputs.push_back(it->second);
        }
      }
      sop.inputs = std::move(inputs);
      const size_t out = stmt_output_.at(stmt);
      sop.outputs.push_back(out);
      problem.ops.push_back(std::move(sop));
      op_labels.push_back(call->op_->name_);
      op_stmts.push_back(stmt);
    }
  }

 private:
  std::unordered_map<const Var*, size_t> tensor_index_;
  std::unordered_map<const Stmt*, size_t> stmt_output_;
  std::unordered_set<const Var*> in_params_;
  bool declined_ = false;

  size_t TensorId(const VarPtr& var) {
    const Var* raw = var.get();
    auto it = tensor_index_.find(raw);
    if (it != tensor_index_.end()) {
      return it->second;
    }
    auto tt = As<TensorType>(var->GetType());
    int64_t w = 1;
    int64_t h = 1;
    // Out of scope for v0 (non-tensor compute output, or a dynamic/symbolic shape): mark the
    // function declined rather than CHECK-crashing. A placeholder tensor still gets registered
    // so tensor indices stay consistent for the remainder of the (now-discarded) build.
    if (tt == nullptr || !ShapeWH(tt, &w, &h)) declined_ = true;
    const size_t idx = problem.tensors.size();
    problem.tensors.push_back(::Tensor{w, h, MapSolverDType(tt != nullptr ? tt->dtype_ : DataType::FP32)});
    tensor_index_[raw] = idx;
    return idx;
  }

  // Returns false (and leaves *w,*h at a safe placeholder) if any dim is dynamic/symbolic.
  static bool ShapeWH(const TensorTypePtr& tt, int64_t* w, int64_t* h) {
    const auto& shape = tt->shape_;
    bool ok = true;
    auto dim = [&](size_t i) -> int64_t {
      auto ci = As<ConstInt>(shape[i]);
      if (ci == nullptr) {  // dynamic/symbolic -> out of scope for v0
        ok = false;
        return 1;
      }
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
    return ok;
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
  f << "],\n  \"dtypes\": [";
  for (size_t i = 0; i < nt; ++i) {
    const char* s = "FP32";
    switch (p.tensors[i].dtype) {
      case ::DType::FP16:  s = "FP16";  break;
      case ::DType::BF16:  s = "BF16";  break;
      case ::DType::INT32: s = "INT32"; break;
      case ::DType::INT16: s = "INT16"; break;
      case ::DType::INT8:  s = "INT8";  break;
      case ::DType::BOOL:  s = "BOOL";  break;
      case ::DType::FP32:  s = "FP32";  break;
    }
    f << (i ? "," : "") << "\"" << s << "\"";
  }
  f << "],\n  \"op_types\": [";
  for (size_t i = 0; i < no; ++i)
    f << (i ? "," : "") << (p.ops[i].type == ::OpType::MatMul ? "\"MatMul\"" : "\"Pointwise\"");
  f << "],\n  \"fast_memory_capacity\": " << p.fast_memory_capacity;
  // 910B topology + grounded pto-isa machine model — emit so a dumped instance
  // re-loads (io.cpp) into the SAME grounded cost path the pass solved with.
  f << ",\n  \"num_cube_cores\": " << p.num_cube_cores
    << ",\n  \"num_vector_cores\": " << p.num_vector_cores
    << ",\n  \"cube_capacity\": " << p.cube_capacity
    << ",\n  \"vec_capacity\": " << p.vec_capacity
    << ",\n  \"l1_capacity\": " << p.l1_capacity
    << ",\n  \"cube_compute_cost\": " << p.cube_compute_cost
    << ",\n  \"kernel_fill_cost\": " << p.kernel_fill_cost
    << ",\n  \"cube_freq_hz\": " << p.cube_freq_hz
    << ",\n  \"bw_gm_l1\": " << p.bw_gm_l1
    << ",\n  \"bw_l0c_gm\": " << p.bw_l0c_gm
    << ",\n  \"bw_l1_l0a\": " << p.bw_l1_l0a
    << ",\n  \"bw_l1_l0b\": " << p.bw_l1_l0b
    << ",\n  \"bw_gm_ub\": " << p.bw_gm_ub
    << ",\n  \"bw_ub_gm\": " << p.bw_ub_gm
    << ",\n  \"hbm_aggregate_gibps\": " << p.hbm_aggregate_gibps
    << ",\n  \"l0_tile_m\": " << p.l0_tile_m
    << ",\n  \"l0_tile_n\": " << p.l0_tile_n
    << ",\n  \"vec_reg_bytes\": " << p.vec_reg_bytes
    << ",\n  \"vec_op_head\": " << p.vec_op_head
    << ",\n  \"vec_op_tail\": " << p.vec_op_tail
    << ",\n  \"vec_slope_pw\": " << p.vec_slope_pw
    << ",\n  \"vec_slope_reduce\": " << p.vec_slope_reduce << "\n}\n";
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
  int64_t split = 1;  // parallel split-K factor S (cores ganged per spatial tile; the
                      // S partials over K/S each are atomic-added). 1 = no split-K.
  // Solver spatial grid region COUNTS (TileConfig::parts_m/parts_n). 0 => UNSET:
  // w/h are exact divisors (the legacy uniform-tile path). >0 => the solver chose a
  // parts_m x parts_n grid whose region extents differ by <=1 fractal per axis, and
  // w/h then carry the MAX (physical) region extent (types.h:180-194, partition_axis).
  // Threaded so the emitter can DETECT a non-uniform grid: PyPTO reconstructs a
  // ceil(M/h) x ceil(N/w) grid, which diverges from the solver's balanced partition
  // when the axis is non-uniform. The generic matmul rule floors that grid (under-
  // covers the tail => wrong result) so it Tier-B-declines a non-uniform grid; the
  // vector rule's ceil+clamp overlap stays numerically correct (idempotent, D3).
  int64_t parts_m = 0;
  int64_t parts_n = 0;
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
                                            sp, ForKind::Pipeline, std::move(loop_attrs));
  return {acc_assign, for_stmt};
}

// Distribute a per-tile `body` across the AI cores via SPMD. Replaces the retired
// AutoInCore + chunked-parallel-loop path (upstream #1895, "remove auto_chunk"): each core
// runs ONE tile selected by `tile.get_block_idx()` and assembles its disjoint region into
// the shared output; OutlineIncoreScopes outlines the per-core InCore kernel and propagates
// `core_num`. `t` is the tile-index var the body reads (offset math drives off it); the body
// computes one tile and binds the function output (no IterArg/Yield -- SPMD is data-parallel).
// Emits `tile.get_block_idx` (what the `for i in pl.spmd(...)` parser desugaring produces, so
// the print/parse round-trip is stable); it is index-only and survives ConvertTensorToTileOps.
static StmtPtr SpmdWrap(const VarPtr& t, std::vector<StmtPtr> body, const ExprPtr& count,
                        const std::string& name, const Span& sp) {
  body.insert(body.begin(), std::make_shared<AssignStmt>(
                                t, OpRegistry::GetInstance().Create("tile.get_block_idx", {}, sp), sp));
  // Naming convention of the `for i in pl.spmd(...)` desugaring (ast_parser
  // _split_spmd_for_loop_name_hints): the InCore kernel keeps the base name, the Spmd
  // wrapper gets the `_spmd` suffix -- so the print/parse round-trip stays structurally stable.
  auto kernel = std::make_shared<InCoreScopeStmt>(std::nullopt, name, SeqStmts::Flatten(std::move(body), sp), sp);
  return std::make_shared<SpmdScopeStmt>(count, /*sync_start=*/false, name + "_spmd", kernel, sp);
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
  if (call == nullptr || !IsOp(call, "tensor.matmul") || call->args_.size() != 2) {
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

  // Parallel split-K: the solver ganged S cores per spatial tile (the cost model
  // guarantees S | K/16, so S | K). Split K into S equal slices, each a partial
  // matmul on a separate core, atomic-added into a ZERO-SEEDED output. The
  // (spatial tile, k-slice) pairs form one flat parallel loop of num_m*num_n*S
  // tasks (chunk=1 -> one task submission each); the S slices of a tile atomic-add
  // into the SAME output region, so they merge correctly under concurrency
  // (tensor.assemble atomic=1 -> pto.tstore atomicType=kAdd -> HW SetAtomicAdd).
  if (tile.split > 1 && K % tile.split == 0) {
    const int64_t S = tile.split;
    const int64_t ksz = K / S;
    const int64_t num_tiles = num_m * num_n;
    auto& reg = OpRegistry::GetInstance();
    auto index_type = std::make_shared<ScalarType>(DataType::INDEX);

    // Allocate the output, then ZERO it in a SEPARATE seed kernel: a barrier the S
    // atomic-add partials accumulate onto. TILE the seed across SPMD blocks exactly like
    // the matmul's spatial tiles -- one [h,w] zero tile per block, num_tiles blocks
    // (disjoint -> non-atomic assemble) -- so a large output never materializes a full
    // [M,N] tensor.full in one core's UB (which would overflow: e.g. 256x256 FP32 =
    // 256KB > the 188KB UB budget).
    auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
    auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
    auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);
    auto zero = std::make_shared<ConstFloat>(0.0, dtype, sp);
    // Per-seed-block tile offsets from its block index (SpmdWrap prepends st = get_block_idx()).
    auto st = std::make_shared<Var>(base + "_st", index_type, sp);
    auto s_mi = MakeMul(MakeFloorDiv(st, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
    auto s_ni = MakeMul(MakeFloorMod(st, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
    auto z_call = reg.Create("tensor.full", {MakeIndexTuple({h, w}, sp), zero}, {{"dtype", dtype}}, sp);
    auto z = std::make_shared<Var>(base + "_z", z_call->GetType(), sp);
    auto seed_asm = reg.Create("tensor.assemble", {c_init, z, MakeTuple2(s_mi, s_ni, sp)}, sp);
    auto c_seeded = std::make_shared<Var>(base + "_seeded", seed_asm->GetType(), sp);
    auto seed_scope = SpmdWrap(
        st,
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(z, z_call, sp),
                             std::make_shared<AssignStmt>(c_seeded, seed_asm, sp)},
        MakeIndex(num_tiles, sp), name + "_seed", sp);

    // t in [0, num_tiles*S): ks = t % S (k-slice), sp_idx = t / S (spatial tile) ->
    // mt = sp_idx / num_n, nt = sp_idx % num_n; offsets mi/ni and k_base = ks*ksz.
    auto t = std::make_shared<Var>(base + "_t", index_type, sp);
    auto ks = MakeFloorMod(t, MakeIndex(S, sp), sp);
    auto sp_idx = MakeFloorDiv(t, MakeIndex(S, sp), sp);
    auto mi = MakeMul(MakeFloorDiv(sp_idx, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
    auto ni = MakeMul(MakeFloorMod(sp_idx, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
    auto k_base = MakeMul(ks, MakeIndex(ksz, sp), sp);
    // Per-task partial: pre-slice A/B to this k-slice, then the [h,w] tile matmul over
    // it (k-pipelined within the slice). Pre-slicing keeps BuildTileMatmul unchanged.
    auto a_ks = reg.Create("tensor.slice", {a, MakeIndexTuple({M, ksz}, sp), MakeTuple2(MakeIndex(0, sp), k_base, sp)}, sp);
    auto a_ks_v = std::make_shared<Var>(base + "_aks", a_ks->GetType(), sp);
    auto b_ks = reg.Create("tensor.slice", {b, MakeIndexTuple({ksz, N}, sp), MakeTuple2(k_base, MakeIndex(0, sp), sp)}, sp);
    auto b_ks_v = std::make_shared<Var>(base + "_bks", b_ks->GetType(), sp);
    auto part_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
    auto part = std::make_shared<Var>(base + "_part", part_type, sp);
    std::vector<StmtPtr> body{std::make_shared<AssignStmt>(a_ks_v, a_ks, sp),
                              std::make_shared<AssignStmt>(b_ks_v, b_ks, sp)};
    for (auto& s : BuildTileMatmul(a_ks_v, b_ks_v, mi, ni, h, w, ksz, tile.k, dtype, part, base, sp))
      body.push_back(std::move(s));

    // Atomic-add the partial into the shared output tile -- the S partials per tile merge
    // across SPMD cores. Each core runs ONE (tile, k-slice) selected by get_block_idx; binds c_var.
    auto asm_call =
        reg.Create("tensor.assemble", {ExprPtr(c_seeded), part, MakeTuple2(mi, ni, sp)}, {{"atomic", 1}}, sp);
    body.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));

    auto scope = SpmdWrap(t, std::move(body), MakeIndex(num_tiles * S, sp), name, sp);
    return std::vector<StmtPtr>{c_init_assign, seed_scope, scope};
  }

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

  // A SINGLE flat parallel loop over the num_m*num_n output tiles: t in
  // [0, num_m*num_n), with element offsets mi = (t / num_n)*h, ni = (t % num_n)*w.
  // chunk=1 makes each tile one chunk, so SplitChunkedLoops emits the OUTER
  // (per-tile) loop into the orchestration as N task submissions of one kernel and
  // the INNER (trip 1) as the per-tile kernel — distributed across cores (chunk =
  // tile-count would collapse the OUTER to trip-1, serializing all tiles on one
  // core). The loop is flattened to 1D (not nested 2D) so the orchestration has a
  // single loop var; nested chunk-outer loops collide in the orchestration codegen's
  // variable naming.
  auto t = std::make_shared<Var>(base + "_t", index_type, sp);
  auto mi = MakeMul(MakeFloorDiv(t, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
  auto ni = MakeMul(MakeFloorMod(t, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
  // Per-tile body: compute the [h,w] tile (k-pipeline) and assemble it into the shared
  // output. Each SPMD core runs ONE tile (selected by get_block_idx, prepended by SpmdWrap)
  // and writes its disjoint [h,w] region, binding c_var (no IterArg/Yield -- data-parallel).
  auto tile_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
  auto tile_var = std::make_shared<Var>(base + "_tile", tile_type, sp);
  std::vector<StmtPtr> body_stmts = BuildTileMatmul(a, b, mi, ni, h, w, K, tile.k, dtype, tile_var, base, sp);
  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_init), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  body_stmts.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));

  auto scope = SpmdWrap(t, std::move(body_stmts), MakeIndex(num_m * num_n, sp), name, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// Realize the solver's [w,h] tiling for a RUN of fused POINTWISE ops (1+ ops in
// one group; later ops consume earlier ones' outputs): tile the group output into
// [w,h] regions distributed across the VECTOR cores via the standard AutoInCore
// chunked-parallel path. Each tile's body REPLAYS the whole op chain on [h,w]
// slices — external inputs are sliced (cached per input), the intermediates stay
// on-chip (the fusion), and each op is re-created so its result type is re-inferred
// — then assembles the group output. nullopt if not eligible: a non-pointwise op,
// more than one live-out (a fused group keeps its intermediates internal), a
// non-[M,N]/non-scalar operand (e.g. broadcast), or the whole output being one
// tile (the plain InCore scope handles that). TODO: share the chunked-parallel
// wrapper with TileMatmul.
std::optional<std::vector<StmtPtr>> TilePointwiseGroup(const std::vector<StmtPtr>& run, SolverTile tile,
                                                       const std::string& name) {
  // 1. Every stmt must be `var = <pointwise call>`.
  std::vector<AssignStmtPtr> ops;
  ops.reserve(run.size());
  for (const StmtPtr& s : run) {
    auto a = As<AssignStmt>(s);
    if (a == nullptr) return std::nullopt;
    auto c = As<Call>(a->value_);
    if (c == nullptr || c->op_ == nullptr || ClassifyOp(c) != ::OpType::Pointwise) return std::nullopt;
    ops.push_back(a);
  }
  if (ops.empty()) return std::nullopt;

  // 2. The group output is the single run-var not consumed within the run (a fused
  //    group keeps its intermediates internal). More than one live-out -> bail.
  std::unordered_set<const Var*> defined;
  for (const auto& a : ops) defined.insert(a->var_.get());
  std::unordered_set<const Var*> used_within;
  for (const auto& a : ops) {
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr && defined.count(v.get()) != 0) used_within.insert(v.get());
    }
  }
  AssignStmtPtr out_stmt = nullptr;
  for (const auto& a : ops) {
    if (used_within.count(a->var_.get()) == 0) {
      if (out_stmt != nullptr) return std::nullopt;  // >1 live-out
      out_stmt = a;
    }
  }
  if (out_stmt == nullptr) return std::nullopt;

  const VarPtr c_var = out_stmt->var_;
  auto ct = As<TensorType>(c_var->GetType());
  if (ct == nullptr) return std::nullopt;
  const DataType dtype = ct->dtype_;
  const auto [M, N] = Static2DShape(c_var->GetType());
  if (M < 0) return std::nullopt;

  // 3. Every operand must be an intermediate, an [M,N] external input, or a scalar
  //    (non-2D). A differently-shaped tensor operand (e.g. broadcast) is not handled
  //    by the simple [h,w] slice -> bail.
  for (const auto& a : ops) {
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr && defined.count(v.get()) != 0) continue;  // intermediate
      const auto [aM, aN] = Static2DShape(arg->GetType());
      if (aM < 0) continue;                       // scalar / non-2D -> kept as-is
      if (aM == M && aN == N) continue;           // [M,N] external input -> sliced
      return std::nullopt;                        // other 2D shape -> not tileable here
    }
  }

  int64_t h = (tile.h > 0 && tile.h < M) ? tile.h : M;
  int64_t w = (tile.w > 0 && tile.w < N) ? tile.w : N;
  if (M % h != 0 || N % w != 0) return std::nullopt;
  const int64_t num_m = M / h, num_n = N / w;
  if (num_m == 1 && num_n == 1) {
    return std::nullopt;  // whole output is one tile -> the plain InCore scope handles it
  }

  const Span sp = out_stmt->span_;
  const std::string base = c_var->name_hint_;
  auto& reg = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);

  auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
  auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);
  // A single flat parallel loop over the num_m*num_n tiles (see TileMatmul): t in
  // [0, num_m*num_n), offsets mi = (t / num_n)*h, ni = (t % num_n)*w. chunk=1 -> one
  // task submission per tile; 1D (not nested) avoids the orchestration codegen's
  // nested-loop variable-name collision.
  auto t = std::make_shared<Var>(base + "_t", index_type, sp);
  auto mi = MakeMul(MakeFloorDiv(t, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
  auto ni = MakeMul(MakeFloorMod(t, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);

  // Per-tile body: replay the op chain at element offset [mi,ni]. Each op's operands
  // are mapped to tile-shaped values — an intermediate uses its on-chip tile result,
  // an [M,N] external input is sliced [h,w] (cached per input var), a scalar is kept.
  // Each op is re-created (not copied) so its result type is re-inferred. The group
  // output op writes `tile_var`.
  std::vector<StmtPtr> body_stmts;
  std::unordered_map<const Var*, VarPtr> tilemap;   // intermediate orig var -> tile result
  std::unordered_map<const Var*, VarPtr> slicemap;  // external input var -> its [h,w] slice
  VarPtr tile_var;
  for (const auto& a : ops) {
    auto c = As<Call>(a->value_);
    std::vector<ExprPtr> targs;
    for (const ExprPtr& arg : c->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr) {
        auto it = tilemap.find(v.get());
        if (it != tilemap.end()) {
          targs.push_back(it->second);
          continue;
        }
      }
      const auto [aM, aN] = Static2DShape(arg->GetType());
      if (aM == M && aN == N) {  // external input -> slice (cached per input var)
        if (v != nullptr) {
          auto sit = slicemap.find(v.get());
          if (sit != slicemap.end()) {
            targs.push_back(sit->second);
            continue;
          }
        }
        auto sl = reg.Create("tensor.slice", {arg, MakeIndexTuple({h, w}, sp), MakeTuple2(mi, ni, sp)}, sp);
        auto sv = std::make_shared<Var>(base + "_in", sl->GetType(), sp);
        body_stmts.push_back(std::make_shared<AssignStmt>(sv, sl, sp));
        if (v != nullptr) slicemap[v.get()] = sv;
        targs.push_back(sv);
      } else {
        targs.push_back(arg);  // scalar / non-2D -> as-is
      }
    }
    auto pw = reg.Create(c->op_->name_, targs, c->kwargs_, sp);
    auto res = std::make_shared<Var>(base + (a == out_stmt ? "_tile" : "_t"), pw->GetType(), sp);
    body_stmts.push_back(std::make_shared<AssignStmt>(res, pw, sp));
    tilemap[a->var_.get()] = res;
    if (a == out_stmt) tile_var = res;
  }

  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_init), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  body_stmts.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));

  auto scope = SpmdWrap(t, std::move(body_stmts), MakeIndex(num_m * num_n, sp), name, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// ============================================================================
// Generic tile-and-fuse driver (behind the PYPTO_AUTOFUSE_GENERIC_EMIT flag).
//
// Replacement-in-progress for the per-shape tilers (TileMatmul / TileChainedMatmul /
// TilePointwiseGroup): ONE driver walks a fused group in plan order and applies a
// per-op-class TilingRule, materializing intermediates on-chip. v1 scope:
// single-pinned-axis, single-sink, engine-homogeneous groups. See the design doc
// "Generic tile-and-fuse emitter" for the full SR/A/S contract.
//
// INCREMENTS 1-2: the ELEMENTWISE rule (identity slice-and-replay) + the REDUCTION rule
// (pin the reduced axis full, tile the free axis — the same slice-and-replay, since the
// solver pins the reduced axis and reductions reduce their full sliced axis on-core).
// This unlocks softmax. A group with a MatMul returns nullopt (increment 3, TODO) so the
// caller falls back to the legacy tiler; the flag defaults OFF, so production is
// byte-for-byte unchanged.
// ============================================================================
static bool GenericEmitEnabled() {
  // Re-read per call (not cached) so a test can toggle the flag in-process — the
  // golden net runs every case both flag-off (legacy) and flag-on (driver) to diff
  // them. Cost is negligible (a getenv per fused group). The env var is stable within
  // any one compile, so re-reading never changes behavior mid-compilation.
  const char* v = std::getenv("PYPTO_AUTOFUSE_GENERIC_EMIT");
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// Strict mode turns Tier-B declines (below) into hard failures. OFF in production
// (Tier-B just warns + falls back to legacy); ON in CI/tests so the bake window
// SURFACES illegal-plan conditions instead of silently masking them behind the
// legacy fallback. Set PYPTO_AUTOFUSE_STRICT=1 in the differential/instrumentation
// tests.
static bool GenericStrict() {
  const char* v = std::getenv("PYPTO_AUTOFUSE_STRICT");
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// The generic emitter declines a group in two very different situations, and
// collapsing both into a bare `return std::nullopt` masks solver bugs during the
// dark launch (the window whose whole point is to find them before the legacy net
// is deleted). So split them:
//
//   Tier-A (capability decline) — "not my scope yet": a matmul chain, a broadcast
//     operand, a single-tile output, a dynamic shape. These fire constantly on
//     normal workloads; falling back silently is correct. Call sites just
//     `return std::nullopt;` (optionally a debug counter).
//
//   Tier-B (suspected illegal plan) — "this should be impossible if the solver is
//     correct": a mis-pinned reduction axis (partial reduction), a non-dividing
//     split, an unexpected multi-sink group, a non-uniform grid the floor
//     reconstruction can't cover, a cross-engine group. These are exactly the
//     A2/A3/A4/A7/SR7 conditions the assert list was designed to catch. GenericDeclineB
//     warns (a greppable, metric-able line) and, under strict, fails loudly — then
//     returns std::nullopt so production still falls back to legacy.
//
// `span` is a Span value (safe to evaluate on failure), so INTERNAL_CHECK_SPAN is OK.
static std::nullopt_t GenericDeclineB(const std::string& reason, const Span& span) {
  LOG_WARN << "AutoFuse[generic] TIER-B decline (suspected illegal plan): " << reason;
  if (GenericStrict()) {
    INTERNAL_CHECK_SPAN(false, span)
        << "AutoFuse generic emit TIER-B: " << reason
        << " — the solver produced a plan the v1 emitter contract forbids "
           "(PYPTO_AUTOFUSE_STRICT is on).";
  }
  return std::nullopt;
}

// A CAPABILITY decline — distinct from both Tier-A (silent "not my scope") and Tier-B
// (illegal plan). The solver LEGITIMATELY produced this plan, but a v1 emitter rule to
// realize it faithfully is deferred, so we fall back to a correct-but-lower-fidelity
// path (typically an untiled InCore scope, ignoring the solver's parallel grid). This
// is NOT a correctness bug — the fallback computes the right values — but it IS a
// fidelity gap (the costed parallel schedule is not realized), so it is worth SEEING.
// Logged (not silent) so the bake window can measure how often the solver's grid is
// dropped, feeding the priority of the deferred rule; NEVER asserts (it is expected,
// and common — e.g. non-uniform parts_m/parts_n grids), so strict mode does not abort.
static std::nullopt_t GenericDeclineCap(const std::string& reason) {
  LOG_INFO << "AutoFuse[generic] CAPABILITY decline (plan not faithfully tiled, runs "
              "lower-fidelity fallback): " << reason;
  return std::nullopt;
}

std::optional<std::vector<StmtPtr>> EmitFusedGroupGeneric(const std::vector<StmtPtr>& run,
                                                          SolverTile tile, const std::string& name) {
  auto& reg = OpRegistry::GetInstance();

  // A1 (classify allowlist). Increments 1-2 support ELEMENTWISE + REDUCTION. A MatMul
  // member is out of scope until increment 3 (its rule is TODO); any other class
  // (transform / opaque / position-dependent) is permanently REJECT — both -> nullopt so
  // the caller falls back to the legacy tiler.
  std::vector<AssignStmtPtr> ops;
  ops.reserve(run.size());
  bool has_reduction = false;
  for (const StmtPtr& s : run) {
    auto a = As<AssignStmt>(s);
    if (a == nullptr) return std::nullopt;                          // Tier-A: non-assign in run (capability)
    auto c = As<Call>(a->value_);
    if (c == nullptr || c->op_ == nullptr) return std::nullopt;     // Tier-A: non-call value (capability)
    const ::OpType cls = ClassifyOp(c);
    // Tier-A: a non-{Pointwise,Reduction} op in the run -> fall back. NB a MatMul can land
    // here legitimately — a lone matmul the tiler declined (e.g. a non-uniform grid it runs
    // untiled) is pushed into `run` and reaches flush(); that is NOT a cross-engine group,
    // so it must NOT Tier-B here. Real mixed-group detection is the engine-homogeneity guard
    // at the import boundary (A2/S1), which classifies the solver's group members directly.
    if (cls != ::OpType::Pointwise && cls != ::OpType::Reduction)
      return std::nullopt;
    if (cls == ::OpType::Reduction) has_reduction = true;
    ops.push_back(a);
  }
  if (ops.empty()) return std::nullopt;

  // A7 (single sink): the group output is the single run-var not consumed within the run
  // (a fused group keeps its intermediates on-chip). >1 live-out = multi-output -> S5.
  std::unordered_set<const Var*> defined;
  for (const auto& a : ops) defined.insert(a->var_.get());
  std::unordered_set<const Var*> used_within;
  for (const auto& a : ops)
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr && defined.count(v.get()) != 0) used_within.insert(v.get());
    }
  // Sinks = the group's live-outs (the solver's boundary outputs), in execution order.
  // A fused group MAY have >1 sink (the solver merges sinks that share input data — the
  // point of the fusion). We assemble each into its own output buffer; the shared-input
  // serialization falls out of replaying the ops in the solver's execution order.
  std::vector<AssignStmtPtr> sinks;
  for (const auto& a : ops)
    if (used_within.count(a->var_.get()) == 0) sinks.push_back(a);
  // Tier-B: a group with NO live-out is structurally impossible in SSA (every group has
  // an output; a cycle among run ops cannot exist). If it happens the run/group mapping
  // is corrupt -> surface it.
  if (sinks.empty())
    return GenericDeclineB("group has no live-out (corrupt run/group mapping)", ops.front()->span_);
  AssignStmtPtr out_stmt = sinks.back();  // primary sink (last in exec order): shape/base for
                                          // the single-sink pipeline/split/serial paths

  // S2 (reduction-sink split): the solver may gang S cores per spatial tile, each reducing
  // a SLICE of the reduced axis, the S partials atomic-merged. Realized below for a SUM
  // col-reduction sink (the only lowered AtomicType is kAdd, and emit_strip slices rows =
  // the reduced axis of a col reduction). Any other split>1 (max/min reduction, row-
  // reduction split, reduction-feeds-pointwise) falls through to the NON-split body, which
  // computes the correct values at split=1 — correct, just without the costed parallelism.

  const VarPtr c_var = out_stmt->var_;
  auto ct = As<TensorType>(c_var->GetType());
  if (ct == nullptr) return std::nullopt;
  const DataType dtype = ct->dtype_;
  const auto [M, N] = Static2DShape(c_var->GetType());  // SINK shape (for the output buffer)
  if (M < 0) return std::nullopt;

  // Iteration space = the reference frame for tiling: the MAX extent over every op's
  // input and output shapes, NOT the sink shape. A reduced sink ([M,1]/[1,N]) is
  // smaller than the pre-reduction working shape; tiling must run over the working
  // shape (IM,IN) and pin the reduced axis, while the sink is assembled at its own
  // (reduced) shape. For a plain [M,N] sink IM,IN == M,N, so this is a no-op there.
  int64_t IM = M, IN = N;
  for (const auto& a : ops) {
    const auto [oM, oN] = Static2DShape(a->var_->GetType());
    IM = std::max(IM, oM);
    IN = std::max(IN, oN);
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      const auto [aM, aN] = Static2DShape(arg->GetType());
      IM = std::max(IM, aM);
      IN = std::max(IN, aN);
    }
  }

  // Operand map is IDENTITY over the iteration space: every operand is an intermediate,
  // an [IM,IN] external input (sliced), or a scalar. A differently-shaped 2D external
  // operand (broadcast) is not identity -> out of scope (a Broadcast rule is TODO).
  for (const auto& a : ops)
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr && defined.count(v.get()) != 0) continue;
      const auto [aM, aN] = Static2DShape(arg->GetType());
      if (aM < 0) {
        // Static2DShape returns {-1,-1} for a true scalar AND for a non-2D / dynamic-shape
        // TENSOR alike. Only the former (a non-tensor operand — e.g. a broadcast scale) is
        // carried through as-is; a rank!=2 or symbolic tensor is out of scope for the 2D emit,
        // so DECLINE rather than misclassify it as a scalar and slice it as [IM,IN].
        if (As<TensorType>(arg->GetType()) != nullptr) return std::nullopt;
        continue;                               // true scalar -> kept as-is
      }
      if (aM == IM && aN == IN) continue;       // full external input -> sliced
      return std::nullopt;                      // other 2D shape (broadcast) -> TODO
    }

  // Grid: single shared parallel tile space; pure elementwise has NO pinned axis, so we
  // tile all output axes. R1 ragged: SPMD compiles ONE [h,w] body for all blocks, so a
  // non-dividing free axis uses a CEIL grid with the tail block's offset CLAMPED in-bounds
  // (mi <= M-h). The tail overlaps the previous tile and recomputes it, but the assemble
  // here is NON-ATOMIC (elementwise/reduction — no split-K), so the overlap is idempotent
  // (same input -> same output) and correct without masking. (Split-K matmul cannot use
  // this: atomic-add would double-count the overlap; that path keeps exact division.)
  // Pinned (reduced) axes: a col reduction reduces M, a row reduction reduces N. The
  // reduced axis must span its FULL iteration extent (a per-tile partial reduction is
  // wrong), so PIN it to IM/IN — the solver's per-axis tile value on a reduced axis is
  // the OUTPUT extent (1), not a tile size, so it can't be used there. Only the FREE
  // axis takes the solver's tile.
  bool pin_m = false, pin_n = false;
  if (has_reduction) {
    for (const auto& a : ops) {
      auto rc = As<Call>(a->value_);
      if (ClassifyOp(rc) != ::OpType::Reduction || rc->args_.empty()) continue;
      const auto [riM, riN] = Static2DShape(rc->args_[0]->GetType());
      const auto [roM, roN] = Static2DShape(a->var_->GetType());
      if (riN > 1 && roN == 1) pin_n = true;  // row reduction -> N pinned
      if (riM > 1 && roM == 1) pin_m = true;  // col reduction -> M pinned
    }
  }
  int64_t h = pin_m ? IM : ((tile.h > 0 && tile.h < IM) ? tile.h : IM);
  int64_t w = pin_n ? IN : ((tile.w > 0 && tile.w < IN) ? tile.w : IN);
  const int64_t num_m = (IM + h - 1) / h, num_n = (IN + w - 1) / w;  // ceil
  // A single spatial tile is left to the legacy tiler — UNLESS the solver split the reduced
  // axis (tile.split>1): then the S2 path below still needs to run (the split is what fills the
  // cores, and for a large reduced axis it is what makes the per-core slice fit UB).
  if (num_m == 1 && num_n == 1 && tile.split <= 1) return std::nullopt;

  // A4 (reduction rule): the reduced axis must be PINNED FULL in the tile, else a
  // per-tile reduction would cover only part of the axis (partial reduction = wrong
  // result). Derive each reduction's reduced axis from its input->output collapse and
  // require the tile spans it fully. The solver pins it (grid num=1 on the reduced axis);
  // if a plan somehow does not, fall back to the legacy scope rather than emit a partial
  // reduction. A group whose reductions disagree on the axis reduces both -> both must be
  // full -> num_m==num_n==1 -> already bailed above (matches the cost model's reject).
  if (has_reduction) {
    for (const auto& a : ops) {
      auto c = As<Call>(a->value_);
      if (ClassifyOp(c) != ::OpType::Reduction || c->args_.empty()) continue;
      const auto [iM, iN] = Static2DShape(c->args_[0]->GetType());
      const auto [oM, oN] = Static2DShape(a->var_->GetType());
      if (iM < 0 || oM < 0) return std::nullopt;  // Tier-A: dynamic shape (capability)
      const bool reduces_N = (iN > 1 && oN == 1);   // row reduction [M,N] -> [M,1]
      const bool reduces_M = (iM > 1 && oM == 1);   // col reduction [M,N] -> [1,N]
      // Tier-B (A4): the reduced axis is NOT pinned full -> a per-tile reduction covers
      // only part of the axis = a partial (wrong) reduction. The solver pins it (grid
      // num=1 on the reduced axis); a plan that does not is illegal for v1.
      if (reduces_N && w != IN)
        return GenericDeclineB("reduction's reduced axis N not pinned full (partial reduction, A4)", a->span_);
      if (reduces_M && h != IM)
        return GenericDeclineB("reduction's reduced axis M not pinned full (partial reduction, A4)", a->span_);
      // Tier-B: a Reduction-classed op whose shape is neither a row nor a col collapse =
      // classifier/plan inconsistency.
      if (!reduces_N && !reduces_M)
        return GenericDeclineB("Reduction op with unexpected shape (not a row/col collapse)", a->span_);
    }
  }

  const Span sp = out_stmt->span_;
  const std::string base = c_var->name_hint_;
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);

  auto c_init_call = reg.Create("tensor.create", {MakeIndexTuple({M, N}, sp)}, {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
  auto c_init = std::make_shared<Var>(base + "_out", c_init_call->GetType(), sp);
  auto c_init_assign = std::make_shared<AssignStmt>(c_init, c_init_call, sp);

  // Flat SPMD grid over the num_m*num_n free-axis tiles; block -> (mi,ni). Elementwise
  // never split-Ks, so this decode is spatial-only (the MatMul rule adds the k_slice).
  // On a ragged axis, clamp the offset to <= extent-tile so the last (ceil) block stays
  // in-bounds (overlapping, idempotent per above). Divisible axes get no clamp, so their
  // emit is byte-identical to before.
  auto t = std::make_shared<Var>(base + "_t", index_type, sp);
  ExprPtr mi = MakeMul(MakeFloorDiv(t, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
  ExprPtr ni = MakeMul(MakeFloorMod(t, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
  if (IM % h != 0) mi = MakeMin(mi, MakeIndex(IM - h, sp), sp);
  if (IN % w != 0) ni = MakeMin(ni, MakeIndex(IN - w, sp), sp);

  // Vector DMA-block granule (elements): a vector (none_box) tile's contiguous-axis byte
  // extent must be a multiple of GetVectorDmaAlignmentBytes() (32). Pad each ragged tile
  // axis up to this granule in the ALLOCATED shape while keeping the valid/compute extent
  // ragged (the tile.load 4th arg). Use the MAX granule over the group's dtypes (= smallest
  // dtype_bytes) since type inference forces every tile in the chain to share the padded
  // extent: a mixed FP16/FP32 chain must satisfy the FP16 16-element block, which also
  // satisfies FP32's 8-element one. FP32 -> 8, FP16 -> 16.
  const auto* pctx = PassContext::Current();
  const auto* handler = pctx ? pctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK(handler) << "Internal error: BackendHandler is null in AutoFuse generic emit";
  int64_t min_dtype_bits = static_cast<int64_t>(dtype.GetBit());
  for (const auto& a : ops) {
    if (auto tt = As<TensorType>(a->var_->GetType()))
      min_dtype_bits = std::min(min_dtype_bits, static_cast<int64_t>(tt->dtype_.GetBit()));
    for (const ExprPtr& arg : As<Call>(a->value_)->args_)
      if (auto att = As<TensorType>(arg->GetType()))
        min_dtype_bits = std::min(min_dtype_bits, static_cast<int64_t>(att->dtype_.GetBit()));
  }
  const int64_t g = std::max<int64_t>(1, handler->GetVectorDmaAlignmentBytes() / ((min_dtype_bits + 7) / 8));

  // Reduced-axis padding is now ALLOWED. The original §4.4 concern — a reduction over a ragged
  // reduced axis pads the reduced axis, leaving uninitialized lanes that feed the sum — is
  // resolved: a device experiment on Ascend 910B proved pto.trowsum / pto.tcolsum bound the
  // reduction by the tile's `valid` extent, not the physical (padded) extent (a poison value in
  // the padded lanes is excluded from the result). So the padded reduced-axis lanes cannot
  // corrupt the valid output; the same axis-padding machinery below handles the reduced axis
  // like any free axis, and `valid` (propagated through tile.load/row_sum) bounds the reduction.
  // (Proven for SUM reductions; MAX/MIN use the same valid mechanism and are confirmed by a
  // device row_max probe — see tests/st/runtime/ops/test_auto_fuse_device.py.)

  // Padded ALLOCATED tile extent; valid stays [h,w]. Padding is a no-op on already-aligned
  // axes (AlignUp(x,g)==x), so aligned shapes emit the 3-arg slice byte-identically to before.
  const int64_t h_al = AlignUp(h, g), w_al = AlignUp(w, g);
  const bool tile_ragged = (h_al != h) || (w_al != w);

  // Driver loop: walk the group in order; per op apply the ELEMENTWISE rule — slice each
  // [M,N] input to [h,w] (identity map), read intermediates from on-chip scratch, then
  // reg.Create the op at tile shape (its type-deduction re-infers the tile result), and
  // materialize on-chip. Two caches (§8 unified scratch): `onchip` for intermediates,
  // `input_cache` for external-input slices (dedups DMA under shared coordinates).
  // emit_strip: slice each [M,N] input to a [sh, w] strip at (smi, ni) (padded to the granule with
  // a ragged VALID extent), read intermediates from on-chip scratch, replay each op at strip shape,
  // and return the SINK strip tile. Shared by the serial body and the software-pipeline loop body,
  // so both realize the same slice-and-replay per strip.
  // `onchip` (op var -> its emitted tile) is caller-provided so a MULTI-SINK group can read
  // every sink's tile out after one replay; single-sink callers pass a fresh throwaway map.
  auto emit_strip = [&](int64_t sh, const ExprPtr& smi, const ExprPtr& sni, std::vector<StmtPtr>& out,
                        std::unordered_map<const Var*, VarPtr>& onchip) -> VarPtr {
    // The 32B DMA granule is on the CONTIGUOUS axis only; the other (free) axis has
    // granule 1 (see ascend910b_cost.cpp: "free row axis tiles at 1 element, the
    // contiguous width axis at the 32-byte DMA block"). So pad the row axis ONLY when
    // rows are contiguous — i.e. the tile is col-major, which a reduction group is
    // (softmax/norms). A pure pointwise tile is row-major (cols contiguous) → rows are
    // the FREE axis and need no padding; padding them is the [64,4096] over-fetch that
    // overflows UB. `has_reduction` is an interim proxy for the tile layout, which is
    // not decided until InferTileMemorySpace; the layout-exact version belongs in a
    // post-layout padding pass (which also fixes the legacy tilers — KNOWN_ISSUES).
    const int64_t sh_al = has_reduction ? AlignUp(sh, g) : sh;
    const bool strip_ragged = (sh_al != sh) || (w_al != w);
    onchip.clear();                                      // fresh per replay
    std::unordered_map<const Var*, VarPtr> input_cache;  // external input -> its [sh,w] slice
    VarPtr tv;
    for (const auto& a : ops) {
      auto c = As<Call>(a->value_);
      std::vector<ExprPtr> targs;
      for (const ExprPtr& arg : c->args_) {
        auto v = AsVarLike(arg);
        if (v != nullptr) {
          auto it = onchip.find(v.get());
          if (it != onchip.end()) { targs.push_back(it->second); continue; }  // intermediate on-chip
        }
        const auto [aM, aN] = Static2DShape(arg->GetType());
        if (aM == IM && aN == IN) {  // full external input -> identity [sh,w] slice (cached per input var)
          if (v != nullptr) {
            auto sit = input_cache.find(v.get());
            if (sit != input_cache.end()) { targs.push_back(sit->second); continue; }
          }
          // Pad the ALLOCATED slice to the granule ([sh_al,w_al]) with a ragged VALID extent
          // ([sh,w], the 4th arg); aligned axes -> 3-arg slice (byte-identical). tile.load/store
          // and elementwise/reduction type-inference honor valid.
          ExprPtr sl = strip_ragged
                           ? reg.Create("tensor.slice",
                                        {arg, MakeIndexTuple({sh_al, w_al}, sp), MakeTuple2(smi, sni, sp),
                                         MakeIndexTuple({sh, w}, sp)},
                                        sp)
                           : reg.Create("tensor.slice", {arg, MakeIndexTuple({sh, w}, sp), MakeTuple2(smi, sni, sp)}, sp);
          // Unique name per distinct external input; a name-based consumer must not collapse >1 input.
          auto sv = std::make_shared<Var>(base + "_in" + std::to_string(input_cache.size()), sl->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(sv, sl, sp));
          if (v != nullptr) input_cache[v.get()] = sv;
          targs.push_back(sv);
        } else {
          targs.push_back(arg);  // scalar / non-2D -> as-is
        }
      }
      auto pw = reg.Create(c->op_->name_, targs, c->kwargs_, sp);
      // Unique name per intermediate (a multi-consumer intermediate must keep a distinct name).
      auto res = std::make_shared<Var>(
          a == out_stmt ? (base + "_tile") : (base + "_t" + std::to_string(onchip.size())),
          pw->GetType(), sp);
      out.push_back(std::make_shared<AssignStmt>(res, pw, sp));
      onchip[a->var_.get()] = res;
      if (a == out_stmt) tv = res;
    }
    return tv;
  };

  // MULTI-SINK: a group with >1 live-out (the solver merges sinks that share inputs).
  // Serial for now (pipeline/split are the single-sink refinements): replay the group ONCE
  // in the solver's execution order, then assemble EACH sink into its own output buffer at
  // its (projected) offset. The shared inputs stay resident across sinks precisely because
  // the replay follows the pebbling order — that is the whole point of the merge.
  if (sinks.size() > 1) {
    std::vector<StmtPtr> prologue;  // one tensor.create per sink
    std::vector<StmtPtr> mbody;
    std::unordered_map<const Var*, VarPtr> onchip;
    emit_strip(h, mi, ni, mbody, onchip);  // replay all ops; onchip[var] = every op's tile
    for (const auto& sink : sinks) {
      auto stt = As<TensorType>(sink->var_->GetType());
      if (stt == nullptr) return std::nullopt;  // non-tensor sink -> out of scope
      const auto [sM, sN] = Static2DShape(sink->var_->GetType());
      auto ci_call = reg.Create("tensor.create", {MakeIndexTuple({sM, sN}, sp)},
                                {{"dtype", stt->dtype_}, {"layout", TensorLayout::ND}}, sp);
      auto ci = std::make_shared<Var>(sink->var_->name_hint_ + "_out", ci_call->GetType(), sp);
      prologue.push_back(std::make_shared<AssignStmt>(ci, ci_call, sp));
      auto it = onchip.find(sink->var_.get());
      INTERNAL_CHECK(it != onchip.end()) << "Internal error: multi-sink tile missing for a live-out";
      ExprPtr off_m = (sM < IM) ? MakeIndex(0, sp) : mi;  // reduced axis -> offset 0
      ExprPtr off_n = (sN < IN) ? MakeIndex(0, sp) : ni;
      auto asm_call = reg.Create("tensor.assemble", {ExprPtr(ci), it->second, MakeTuple2(off_m, off_n, sp)}, sp);
      mbody.push_back(std::make_shared<AssignStmt>(sink->var_, asm_call, sp));
    }
    auto scope = SpmdWrap(t, std::move(mbody), MakeIndex(num_m * num_n, sp), name, sp);
    prologue.push_back(scope);
    LOG_INFO << "AutoFuse[generic]: multi-sink group '" << name << "' (" << sinks.size()
             << " live-outs) emitted in execution order";
    return prologue;
  }

  // S2 — SUM col-reduction split. When the solver gangs S cores per free tile (tile.split>1)
  // to parallelize a reduction, partition the reduced M axis across S cores: each reduces its
  // disjoint M-slice to a [1,w] partial, and the S partials ATOMIC-ADD into a zero-seeded [1,N]
  // output (same seed+atomic structure as matmul split-K). Realized ONLY for a clean partition:
  //   - the sink IS a `col_sum` (kAdd is the only lowered AtomicType; emit_strip slices rows =
  //     the reduced axis of a col reduction),
  //   - S == tile.split EXACTLY with IM % (S*g) == 0 (each core's reduced slice rsz=IM/S is a
  //     multiple of the DMA granule g), and
  //   - IN % w == 0 (non-overlapping free tiles).
  // Why these exact-divisibility conditions are REQUIRED, not conservative — under R1 (one SPMD
  // body compiled for every block) a ragged split has NO safe realization when the merge is
  // ATOMIC-ADD:
  //   - Reduced axis, rsz not granule-aligned: emit_strip pads each slice to AlignUp(rsz,g), so
  //     the last slice at offset (S-1)*rsz reads (S-1)*rsz+AlignUp(rsz,g) rows. If rsz % g != 0
  //     that runs PAST IM -> out-of-bounds DMA (and clamping the offset in-bounds instead would
  //     OVERLAP the prior slice -> atomic-add double-counts). Requiring IM % (S*g) == 0 makes
  //     rsz a multiple of g, so AlignUp(rsz,g)==rsz: exactly S disjoint in-bounds slices.
  //     (The elementwise path tolerates a clamp overlap only because its assemble is NON-atomic.)
  //   - Free axis, IN % w != 0: same overlap, same atomic double-count on the tail column band.
  // The costed split is trusted EXACTLY (never rounded to a nearby divisor — that would enlarge
  // each slice and break the solver's UB-fit proof). The solver only costs a realizable S: it
  // draws vector S from divisors of the reduced FRACTAL count (reduced_extent/16) when that axis
  // is 16-aligned (mlsys26 ascend910b_cost.cpp:886, mirroring the matmul kfrac gate id.:870), so
  // IM/S is 16-aligned and this gate holds. The gate is kept as DEFENSE-IN-DEPTH: a non-conforming
  // S from any other cost model declines to the CORRECT non-split body rather than emitting an OOB
  // read. Everything else (max/min, row-reduction split, reduction-feeds-pointwise) also declines.
  if (tile.split > 1 && pin_m && As<Call>(out_stmt->value_) &&
      IsOp(As<Call>(out_stmt->value_), "tensor.col_sum") && IM % (tile.split * g) == 0 && IN % w == 0) {
    const int64_t S = tile.split;               // emit exactly the costed split (no rounding)
    const int64_t rsz = IM / S;                 // disjoint granule-aligned reduced slice (IM % (S*g) == 0)
    auto index_ty = std::make_shared<ScalarType>(DataType::INDEX);
    // Seed: zero each [1,w] output tile (disjoint + non-overlapping since IN % w == 0), tiled over
    // the num_n free tiles, so a large [1,N] output never materializes in one core's UB.
    auto zero = std::make_shared<ConstFloat>(0.0, dtype, sp);
    auto st = std::make_shared<Var>(base + "_st", index_ty, sp);
    ExprPtr s_ni = MakeMul(st, MakeIndex(w, sp), sp);
    auto z_call = reg.Create("tensor.full", {MakeIndexTuple({M, w}, sp), zero}, {{"dtype", dtype}}, sp);
    auto z = std::make_shared<Var>(base + "_z", z_call->GetType(), sp);
    auto seed_asm = reg.Create("tensor.assemble", {c_init, z, MakeTuple2(MakeIndex(0, sp), s_ni, sp)}, sp);
    auto c_seeded = std::make_shared<Var>(base + "_seeded", seed_asm->GetType(), sp);
    auto seed_scope = SpmdWrap(st,
                               std::vector<StmtPtr>{std::make_shared<AssignStmt>(z, z_call, sp),
                                                    std::make_shared<AssignStmt>(c_seeded, seed_asm, sp)},
                               MakeIndex(num_n, sp), name + "_seed", sp);
    // Main grid: t in [0, num_n*S). ks = t % S (M-slice), fidx = t / S (free N tile).
    auto t2 = std::make_shared<Var>(base + "_t", index_ty, sp);
    auto ks = MakeFloorMod(t2, MakeIndex(S, sp), sp);
    auto fidx = MakeFloorDiv(t2, MakeIndex(S, sp), sp);
    ExprPtr r_mi = MakeMul(ks, MakeIndex(rsz, sp), sp);   // disjoint reduced-M slice offset
    ExprPtr sni = MakeMul(fidx, MakeIndex(w, sp), sp);    // free-N tile offset (exact, no overlap)
    std::vector<StmtPtr> sbody;
    std::unordered_map<const Var*, VarPtr> oc_split;
    VarPtr part = emit_strip(rsz, r_mi, sni, sbody, oc_split);  // [1,w] partial col_sum over the M-slice
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_seeded), part, MakeTuple2(MakeIndex(0, sp), sni, sp)},
                               {{"atomic", 1}}, sp);
    sbody.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));
    auto scope = SpmdWrap(t2, std::move(sbody), MakeIndex(num_n * S, sp), name, sp);
    LOG_INFO << "AutoFuse[generic]: SUM col-reduction group '" << name << "' split " << S
             << " ways over the reduced axis (S2, atomic-add merge)";
    return std::vector<StmtPtr>{c_init_assign, seed_scope, scope};
  }

  // Software-pipelining: chunk the FREE axis into strips and emit a ForKind::Pipeline loop threading
  // the output through ONE iter_arg, so LowerPipelineLoops (unroll+tag) + CanonicalizeIOOrder (cluster
  // loads) + MemoryReuse (per-stage ping-pong load buffers) realize the max(compute,ddr) roofline the
  // cost model prices (db_roofline). A reduction pins its reduced axis full, so we chunk the NON-PINNED
  // axis: h (rows) for row-reduction/pointwise. A COL reduction pins h (=M); chunking it would split
  // the reduction -> keep the SERIAL body there (deferred), which the model's db=false path prices as
  // compute+ddr anyway.
  bool has_col_reduction = false;
  if (has_reduction) {
    for (const auto& a : ops) {
      auto c = As<Call>(a->value_);
      if (ClassifyOp(c) != ::OpType::Reduction || c->args_.empty()) continue;
      const auto [ciM, ciN] = Static2DShape(c->args_[0]->GetType());
      const auto [coM, coN] = Static2DShape(a->var_->GetType());
      if (ciM > 1 && coM == 1) has_col_reduction = true;  // reduces height -> pins h
    }
  }
  // Strip count: the largest divisor of h in {8,4,2} (prefer more strips for a steady-state pipeline,
  // trip >= 2*stage) giving EQUAL strips (h % num_strips == 0 -> no ragged-strip clamp; tile-level
  // ragged M is already handled by the mi clamp). Fall back to serial (1 strip) when h can't be
  // chunked >=2 ways or the group has a col reduction.
  int64_t num_strips = 1;
  if (!has_col_reduction) {
    for (int64_t ns : {8, 4, 2}) {
      if (ns <= h && h % ns == 0) { num_strips = ns; break; }
    }
  }

  std::vector<StmtPtr> body_stmts;
  if (num_strips < 2) {
    // Serial (matches the cost model's db=false): the whole tile is one strip.
    std::unordered_map<const Var*, VarPtr> oc_serial;
    VarPtr tv = emit_strip(h, mi, ni, body_stmts, oc_serial);
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_init), tv, MakeTuple2(mi, ni, sp)}, sp);
    body_stmts.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));
  } else {
    // Pipelined: for s in pipeline(0, num_strips, stage=2) iter_args=[out <- c_init]:
    //   strip at (mi + s*strip_h, ni); replay ops -> strip tile; out = assemble(out, strip, off); yield.
    // The output iter_arg is used ONLY as the assemble target (source/offset reference s & t, not the
    // iter_arg), so ConvertTensorToTileOps::RewriteReturnedAssembleLoopToStore lowers it to an in-place
    // tile.store while preserving kind==Pipeline + pipeline_stages.
    const int64_t strip_h = h / num_strips;
    auto index_ty = std::make_shared<ScalarType>(DataType::INDEX);
    auto s = std::make_shared<Var>(base + "_s", index_ty, sp);
    auto out_iter = std::make_shared<IterArg>(base + "_out_it", c_init->GetType(), ExprPtr(c_init), sp);
    ExprPtr smi = MakeAdd(mi, MakeMul(s, MakeIndex(strip_h, sp), sp), sp);
    std::vector<StmtPtr> loop_body;
    std::unordered_map<const Var*, VarPtr> oc_pipe;
    VarPtr tv = emit_strip(strip_h, smi, ni, loop_body, oc_pipe);
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(out_iter), tv, MakeTuple2(smi, ni, sp)}, sp);
    auto out_next = std::make_shared<Var>(base + "_out_n", asm_call->GetType(), sp);
    loop_body.push_back(std::make_shared<AssignStmt>(out_next, asm_call, sp));
    loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{out_next}, sp));
    StmtPtr body = SeqStmts::Flatten(std::move(loop_body), sp);
    std::vector<std::pair<std::string, std::any>> loop_attrs = {{kPipelineStagesAttr, /*stages=*/2}};
    auto for_stmt = std::make_shared<ForStmt>(s, MakeIndex(0, sp), MakeIndex(num_strips, sp), MakeIndex(1, sp),
                                              std::vector<IterArgPtr>{out_iter}, body, std::vector<VarPtr>{c_var},
                                              sp, ForKind::Pipeline, std::move(loop_attrs));
    body_stmts.push_back(for_stmt);
  }

  auto scope = SpmdWrap(t, std::move(body_stmts), MakeIndex(num_m * num_n, sp), name, sp);
  LOG_INFO << "AutoFuse[generic]: " << (has_reduction ? "elementwise+reduction" : "elementwise")
           << " group '" << name << "' tiled by the generic driver (" << ops.size() << " ops, "
           << (num_m * num_n) << " tiles, " << (num_strips >= 2 ? num_strips : 1) << " pipeline strips)";
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// The MATMUL rule (increment 3), for a LONE matmul (chains are S4 -> fall back). Its
// tiling IS TileMatmul (the grid + split-K tiled-seed + BuildTileMatmul k-pipeline); the
// generic path adds the fail-loud A3/SR7 contract asserts the legacy tiler lacks and
// unifies dispatch behind the flag. The tiling body is REUSED for now; absorbing it so
// the legacy TileMatmul dispatch can retire is the parity-prep step.
std::optional<std::vector<StmtPtr>> EmitLoneMatmulGeneric(const AssignStmtPtr& assign,
                                                          SolverTile tile, const std::string& name) {
  auto call = As<Call>(assign->value_);
  if (call == nullptr || call->op_ == nullptr || ClassifyOp(call) != ::OpType::MatMul ||
      call->args_.size() != 2) {
    return std::nullopt;  // A1: not a lone matmul -> fall back (a chain is S4)
  }
  auto ct = As<TensorType>(assign->var_->GetType());
  if (ct == nullptr) return std::nullopt;
  const auto [M, N] = Static2DShape(assign->var_->GetType());
  const auto [lM, lK] = Static2DShape(call->args_[0]->GetType());
  if (M < 0 || lM != M) return std::nullopt;
  const int64_t K = lK;

  // Tier-B — A3 (split legality) + SR7/A3(iv): a split may only fractal-partition the
  // contraction into equal 16-aligned slices -> split | (K/16). The sink IS a matmul
  // (A3(iii)) and the sole atomic target. The solver only picks split ∈ divisors(K/16); a
  // plan that violates it would leave a ragged/misaligned K-slice — surface it rather than
  // emit a wrong split.
  if (tile.split > 1) {
    const int64_t kfrac = K / 16;
    if (K % 16 != 0 || kfrac % tile.split != 0 || K % tile.split != 0)
      return GenericDeclineB("split-K does not fractal-partition K (split ∤ K/16, A3/SR7)", assign->span_);
  }

  // Capability — non-uniform grid: when the solver chose a parts_m/parts_n grid (parts_*>0),
  // w/h are the MAX region extents and the balanced partition has some smaller regions
  // (extents differ by <=1 fractal). The v1 emitter tiles only UNIFORM grids: TileMatmul
  // guards `M%h==0 && N%w==0` (line ~640) and safely declines a non-uniform grid to a
  // single UNTILED InCore scope — correct values, but the solver's parallel parts×split
  // grid is not realized (a fidelity gap, not a correctness bug: there is NO tail-dropping,
  // the guard prevents it). Faithfully realizing the balanced grid needs the AxisPartition
  // decode (deferred, P2). Surface it so we can measure how often it happens; do not abort.
  // parts_*==0 (uniform/ad-hoc tile) => w/h are exact divisors => M%h==0, guard never fires.
  const int64_t mh = (tile.h > 0 && tile.h < M) ? tile.h : M;
  const int64_t mw = (tile.w > 0 && tile.w < N) ? tile.w : N;
  if ((mh < M && M % mh != 0) || (mw < N && N % mw != 0))
    return GenericDeclineCap("non-uniform spatial grid (parts_m=" + std::to_string(tile.parts_m) +
                             ",parts_n=" + std::to_string(tile.parts_n) +
                             "): runs untiled (uniform-grid only in v1, P2 decode deferred)");

  auto tiled = TileMatmul(assign, tile, name);  // MatMul rule body: grid + split-K seed + k-pipeline
  if (!tiled) return std::nullopt;
  LOG_INFO << "AutoFuse[generic]: matmul group '" << name << "' tiled by the generic driver (split="
           << tile.split << ")";
  return tiled;
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
  if (c1 == nullptr || c2 == nullptr || !IsOp(c1, "tensor.matmul") ||
      !IsOp(c2, "tensor.matmul") || c1->args_.size() != 2 || c2->args_.size() != 2) {
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
  // A single flat parallel loop over the num_m*num_n tiles (see TileMatmul): t in
  // [0, num_m*num_n), offsets mi = (t / num_n)*h, ni = (t % num_n)*w. chunk=1 -> one
  // task submission per tile; 1D (not nested) avoids the orchestration codegen's
  // nested-loop variable-name collision.
  auto t = std::make_shared<Var>(base + "_t", index_type, sp);
  auto mi = MakeMul(MakeFloorDiv(t, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
  auto ni = MakeMul(MakeFloorMod(t, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);

  // Per-tile body: the inner serial chain (T_band on-chip), assembled into the output.
  auto tile_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
  auto tile_var = std::make_shared<Var>(base + "_tile", tile_type, sp);
  std::vector<StmtPtr> body_stmts = build_chain(mi, ni, tile_var);
  auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_init), tile_var, MakeTuple2(mi, ni, sp)}, sp);
  body_stmts.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));

  auto scope = SpmdWrap(t, std::move(body_stmts), MakeIndex(num_m * num_n, sp), name, sp);
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// Collect the Vars a flat tensor-op stmt READS. Covers the pre-emit auto_fuse body shapes:
// an AssignStmt of a Call (its args), an SSA rebind (a plain Var value), and a ReturnStmt (its
// values). The marked body is a flat DAG of Calls before outlining — no control flow, no Submit.
void CollectStmtUses(const StmtPtr& s, std::vector<const Var*>* out) {
  auto add = [&](const ExprPtr& e) {
    if (auto v = AsVarLike(e)) out->push_back(v.get());
  };
  if (auto a = As<AssignStmt>(s)) {
    if (auto c = As<Call>(a->value_)) {
      for (const ExprPtr& arg : c->args_) add(arg);
    } else {
      add(a->value_);  // SSA rebind `b = a`
    }
  } else if (auto r = As<ReturnStmt>(s)) {
    for (const ExprPtr& v : r->value_) add(v);
  }
}

// True when some solver group's stmts are NOT a contiguous run in SSA body order — i.e. an op of
// another group (or an ungrouped stmt) lexically separates two of its members. A group occupies
// [min,max] body indices; if it has fewer members than that span is wide, something is interspersed.
bool GroupsAreFragmented(const std::vector<StmtPtr>& body,
                         const std::unordered_map<const Stmt*, size_t>& stmt_group) {
  std::unordered_map<size_t, size_t> gmin, gmax, gcount;
  for (size_t i = 0; i < body.size(); ++i) {
    auto it = stmt_group.find(body[i].get());
    if (it == stmt_group.end()) continue;
    const size_t g = it->second;
    if (gcount.find(g) == gcount.end()) {
      gmin[g] = i;
      gmax[g] = i;
      gcount[g] = 1;
    } else {
      gmax[g] = i;
      gcount[g]++;
    }
  }
  for (const auto& [g, c] : gcount)
    if (gmax[g] - gmin[g] + 1 != c) return true;
  return false;
}

// Reorder a flat body so every solver group is CONTIGUOUS while preserving data dependencies, so
// the contiguous-run emit below realizes each group as ONE fused scope (the on-chip working set the
// cost model priced) instead of fragmenting a non-contiguous group into several scopes that
// round-trip the shared intermediate through DDR. Contract each group to one "unit" (each ungrouped
// stmt is its own unit); the solver's grouping is convex, so the unit graph is a DAG — a topological
// sort clusters each group. Falls back to the original order if a cycle appears (a non-convex
// grouping — the contiguous-run emit then still produces correct, if fragmented, code).
std::vector<StmtPtr> ReorderBodyByGroup(const std::vector<StmtPtr>& body,
                                        const std::unordered_map<const Stmt*, size_t>& stmt_group) {
  if (!GroupsAreFragmented(body, stmt_group)) return body;  // common case: already contiguous
  const size_t n = body.size();
  size_t max_group = 0;
  for (const auto& kv : stmt_group) max_group = std::max(max_group, kv.second);
  const size_t ungrouped_base = max_group + 1;  // ungrouped stmt i -> its own unit key

  // Assign dense unit ids in first-appearance (SSA) order: grouped stmts share their group's unit.
  std::unordered_map<size_t, size_t> unit_index;  // raw key -> dense id
  std::vector<std::vector<size_t>> unit_stmts;     // dense id -> body indices (in SSA order)
  std::vector<size_t> unit_of_stmt(n);
  std::unordered_map<const Var*, size_t> def_unit;  // defined Var -> its unit
  for (size_t i = 0; i < n; ++i) {
    auto git = stmt_group.find(body[i].get());
    const size_t key = (git != stmt_group.end()) ? git->second : ungrouped_base + i;
    auto uit = unit_index.find(key);
    size_t u;
    if (uit == unit_index.end()) {
      u = unit_stmts.size();
      unit_index[key] = u;
      unit_stmts.emplace_back();
    } else {
      u = uit->second;
    }
    unit_stmts[u].push_back(i);
    unit_of_stmt[i] = u;
    if (auto a = As<AssignStmt>(body[i]))
      if (a->var_) def_unit[a->var_.get()] = u;
  }
  const size_t U = unit_stmts.size();

  // Unit dependency graph: unit X depends on unit Y if a stmt in X reads a Var defined in Y.
  std::vector<std::set<size_t>> preds(U);
  std::vector<const Var*> uses;
  for (size_t i = 0; i < n; ++i) {
    uses.clear();
    CollectStmtUses(body[i], &uses);
    for (const Var* v : uses) {
      auto dit = def_unit.find(v);
      if (dit != def_unit.end() && dit->second != unit_of_stmt[i]) preds[unit_of_stmt[i]].insert(dit->second);
    }
  }
  std::vector<size_t> indeg(U, 0);
  std::vector<std::vector<size_t>> succ(U);
  for (size_t u = 0; u < U; ++u)
    for (size_t p : preds[u]) {
      succ[p].push_back(u);
      indeg[u]++;
    }

  // Kahn topological sort; the ready set is ordered so the smallest dense unit id (earliest SSA
  // appearance) wins ties — deterministic, and identity on an already-contiguous body.
  std::set<size_t> ready;
  for (size_t u = 0; u < U; ++u)
    if (indeg[u] == 0) ready.insert(u);
  std::vector<size_t> order;
  order.reserve(U);
  while (!ready.empty()) {
    const size_t u = *ready.begin();
    ready.erase(ready.begin());
    order.push_back(u);
    for (size_t w : succ[u])
      if (--indeg[w] == 0) ready.insert(w);
  }
  if (order.size() != U) return body;  // cycle (non-convex grouping) -> keep original order

  std::vector<StmtPtr> out;
  out.reserve(n);
  for (size_t u : order)
    for (size_t i : unit_stmts[u]) out.push_back(body[i]);
  return out;
}

// Rewrite a function body to realize the solver's decision. A matmul becomes its
// own self-scoped tiled kernel (the solver's `[w,h]` output tiling, an InCore
// kernel per tile, the per-tile k-pipeline inside) emitted at the orchestration
// level; two chained matmuls in one group become a single fused kernel (the
// intermediate stays on-chip, see TileChainedMatmul); every other fused group is
// a maximal *contiguous* run of same-group compute stmts wrapped in one
// InCoreScopeStmt. Non-contiguous groups are first reordered contiguous
// (ReorderBodyByGroup) so each group emits as one scope. The body is in SSA order.
StmtPtr EmitFusedScopes(const StmtPtr& body,
                        const std::unordered_map<const Stmt*, size_t>& stmt_group,
                        const std::unordered_map<const Stmt*, SolverTile>& stmt_tile,
                        const std::unordered_map<const Stmt*, size_t>& stmt_exec) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = As<SeqStmts>(body)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(body);
  }

  // Cluster each solver group into a contiguous run (dependency-preserving) so a group whose
  // members are interleaved with another group's in SSA order still emits as ONE fused scope.
  body_stmts = ReorderBodyByGroup(body_stmts, stmt_group);

  // Detect 2-matmul chains within a group: MM2 = matmul(T, D) where its left
  // operand T is the output of MM1 = matmul(A, B) in the SAME group. Such a pair
  // is emitted as one fused kernel (T on-chip) instead of two separate matmul
  // kernels (T round-tripping DDR). The body is in SSA order, so MM1 precedes MM2.
  auto matmul_call = [](const StmtPtr& s) -> CallPtr {
    auto a = As<AssignStmt>(s);
    if (a == nullptr) return nullptr;
    auto c = As<Call>(a->value_);
    if (c == nullptr || !IsOp(c, "tensor.matmul") || c->args_.size() != 2) {
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
  std::unordered_set<long> flushed_groups;  // DIAGNOSTIC: detect a group emitted as >1 scope
  // Flush the accumulated run. A lone pointwise op gets the solver's [w,h]
  // cross-core tiling (TilePointwise); everything else is wrapped in one InCore
  // scope (multi-op groups, reductions, or pointwise that needs no tiling).
  auto flush = [&]() {
    if (run.empty()) {
      return;
    }
    // ReorderBodyByGroup makes every group contiguous before this loop, so a group should flush
    // exactly once. If it flushes twice, the reorder fell back (a non-convex grouping -> cycle in
    // the unit graph): the group is emitted as multiple scopes, so the cross-fragment intermediate
    // round-trips DDR instead of staying on-chip (correctness preserved: it becomes a materialized
    // live-out; only the on-chip working set the solver costed is lost). Surface it.
    if (run_group >= 0 && !flushed_groups.insert(run_group).second) {
      LOG_INFO << "AutoFuse[generic]: WARNING group " << run_group
               << " emitted as multiple scopes (non-convex grouping -> reorder fell back; fidelity loss)";
    }
    // Replay the group in the solver's EXECUTION ORDER (its depth-first pebbling order),
    // NOT the SSA/body order. That order is what the cost model evaluated the working-set
    // peak against, so it is the only order guaranteed to fit UB — any other valid topo
    // order may exceed it. stable_sort keeps SSA order as the tie-break for ops the solver
    // left mutually unordered (and for any stmt missing from the map, defensively).
    std::stable_sort(run.begin(), run.end(), [&](const StmtPtr& a, const StmtPtr& b) {
      auto ia = stmt_exec.find(a.get());
      auto ib = stmt_exec.find(b.get());
      const size_t pa = ia != stmt_exec.end() ? ia->second : 0;
      const size_t pb = ib != stmt_exec.end() ? ib->second : 0;
      return pa < pb;
    });
    const Span scope_span = run.front()->span_;
    const std::string nm = "fused_" + std::to_string(run_group);
    // A run of fused pointwise ops gets the solver's [w,h] cross-core tiling
    // (TilePointwiseGroup); anything it cannot tile (non-pointwise op, >1 live-out,
    // single-tile output) falls back to one plain InCore scope.
    auto tit = stmt_tile.find(run.front().get());
    if (tit != stmt_tile.end()) {
      // Behind the flag, the generic tile-and-fuse driver gets first refusal; it handles
      // only what its implemented rules cover (increment 1: elementwise) and returns
      // nullopt otherwise, so we fall through to the legacy tiler. Flag off => never called.
      if (GenericEmitEnabled()) {
        if (auto generic = EmitFusedGroupGeneric(run, tit->second, nm)) {
          for (auto& s : *generic) {
            top.push_back(std::move(s));
          }
          run.clear();
          run_group = -1;
          return;
        }
      }
      if (auto tiled = TilePointwiseGroup(run, tit->second, nm)) {
        for (auto& s : *tiled) {
          top.push_back(std::move(s));
        }
        run.clear();
        run_group = -1;
        return;
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
        const std::string nm = "fused_" + std::to_string(g);
        // Behind the flag, the generic MatMul rule gets first refusal (adds the A3/SR7
        // asserts); it returns nullopt if it can't own the group, so we fall back to the
        // legacy tiler. Flag off => never called.
        if (GenericEmitEnabled()) {
          tiled = EmitLoneMatmulGeneric(assign, tit->second, nm);
        }
        if (!tiled) {
          tiled = TileMatmul(assign, tit->second, nm);
        }
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

// ---------------------------------------------------------------------------
// Wire a return-based fused function to a named Out param
// ---------------------------------------------------------------------------
//
// A marked auto_fuse function is return-based (`def f(a) -> Tensor: ...; return
// d`) and the emitters realize its output as a runtime-allocated `c_init =
// tensor.create([M,N])` that the final `tensor.assemble` (serial) / pipeline
// loop-carry (pipelined) writes and the ReturnStmt returns. Orchestration
// codegen only emits an `add_output` write-back for a param the return ALIASES
// (see return_lineage), so a purely-allocated output is written to a throwaway
// buffer — invisible to a by-parameter caller (the device / ST harness binds
// I/O by param position, not by return value; the output buffer stays
// unwritten). Lift the output buffer into an appended `Out` param: the SAME
// `c_init` Var is MOVED from its `tensor.create` binding into the param list,
// so every body reference (assemble arg0 / iter_arg init) still resolves to it,
// the return lineage now lands on a param, and codegen emits the write-back.
// The emit is otherwise byte-identical.
//
// Scoped to the safe, common case (a standalone entry kernel): a single-tensor
// return, no existing Out param, and not called by another function (appending
// a param would break its callsites). Anything else is left return-based.
void MaybeLiftReturnToOutParam(const std::shared_ptr<Function>& func,
                               const std::unordered_set<std::string>& called_funcs) {
  // Already has an output param (user-written) -> nothing to wire.
  for (ParamDirection d : func->param_directions_) {
    if (d == ParamDirection::Out || d == ParamDirection::InOut) return;
  }
  // Called internally: changing the signature would break its callsites.
  if (called_funcs.count(func->name_) != 0) return;

  // Index the body: first ReturnStmt, per-var defining assign, and for-loop
  // carry edges (iter_arg / tensor return_var -> iter_arg init var). Mirrors
  // return_lineage's tracer so the return-buffer trace below matches codegen.
  class Indexer : public IRVisitor {
   public:
    ReturnStmtPtr ret;
    std::unordered_map<const Var*, AssignStmtPtr> var_def;
    std::unordered_map<const Var*, const Var*> carry;

   protected:
    void VisitStmt_(const ReturnStmtPtr& r) override {
      if (!ret) ret = r;
    }
    void VisitStmt_(const AssignStmtPtr& a) override {
      if (a->var_) var_def.emplace(a->var_.get(), a);
      IRVisitor::VisitStmt_(a);
    }
    void VisitStmt_(const ForStmtPtr& f) override {
      for (size_t i = 0; i < f->iter_args_.size(); ++i) {
        auto init = AsVarLike(f->iter_args_[i]->initValue_);
        if (!init) continue;
        carry[f->iter_args_[i].get()] = init.get();
        // Scalar carries may be overwritten by the body; only tensor carries
        // propagate the init (matches return_lineage).
        if (i < f->return_vars_.size() && AsTensorTypeLike(f->return_vars_[i]->GetType())) {
          carry[f->return_vars_[i].get()] = init.get();
        }
      }
      IRVisitor::VisitStmt_(f);
    }
  } idx;
  idx.VisitStmt(func->body_);

  if (!idx.ret || idx.ret->value_.empty()) return;

  // Trace a returned var back to its output buffer's `tensor.create`, through SSA rebinds,
  // tensor.assemble / set_validshape (arg0), tile.store (arg2), and for-loop carries. The
  // create's AssignStmt is that output's `c_init`; returns nullptr if it can't be reached.
  auto trace_to_create = [&](const VarPtr& ret_var) -> AssignStmtPtr {
    const Var* cur = ret_var.get();
    std::unordered_set<const Var*> seen;
    while (cur != nullptr) {
      if (!seen.insert(cur).second) break;
      if (auto it = idx.carry.find(cur); it != idx.carry.end()) { cur = it->second; continue; }
      auto dit = idx.var_def.find(cur);
      if (dit == idx.var_def.end()) break;
      const ExprPtr& val = dit->second->value_;
      if (auto rv = AsVarLike(val)) { cur = rv.get(); continue; }
      auto call = As<Call>(val);
      if (call == nullptr) break;
      if (IsOp(call, "tensor.create")) return dit->second;
      if (IsOp(call, "tensor.assemble") || IsOp(call, "tensor.set_validshape")) {
        auto a0 = !call->args_.empty() ? AsVarLike(call->args_[0]) : nullptr;
        if (!a0) break;
        cur = a0.get();
        continue;
      }
      if (IsOp(call, "tile.store")) {
        auto a2 = call->args_.size() >= 3 ? AsVarLike(call->args_[2]) : nullptr;
        if (!a2) break;
        cur = a2.get();
        continue;
      }
      break;
    }
    return nullptr;
  };

  // Trace EVERY returned tensor to its output buffer's create. All-or-nothing: a partial
  // lift (some returns wired, some not) would leave an inconsistent ABI, so bail on any
  // return we can't wire. Each buffer's create must be a top-level child (safe to remove)
  // and DISTINCT (two returns aliasing one buffer -> bail).
  auto seq = As<SeqStmts>(func->body_);
  if (seq == nullptr) return;
  // Index the top-level children ONCE (O(N)) so the per-return "is this create a top-level
  // child?" test is an O(1) lookup, not an O(N) scan -> the loop is O(R) not O(R*N).
  std::unordered_set<const Stmt*> top_level;
  top_level.reserve(seq->stmts_.size());
  for (const StmtPtr& s : seq->stmts_) top_level.insert(s.get());
  std::vector<AssignStmtPtr> creates;
  std::unordered_set<const Stmt*> create_set;
  for (const ExprPtr& rv : idx.ret->value_) {
    auto ret_var = AsVarLike(rv);
    if (!ret_var) return;
    auto ca = trace_to_create(ret_var);
    if (!ca) return;
    if (top_level.count(ca.get()) == 0) return;       // create not a removable top-level child
    if (!create_set.insert(ca.get()).second) return;  // returns share a buffer -> aliasing, bail
    creates.push_back(ca);
  }

  // Append an Out param per returned buffer (in return order — the harness binds by
  // position), moving each c_init Var from its create into the param list, and drop the
  // now-dead creates. The return lineage rebinds each return to its Out param.
  for (const auto& ca : creates) {
    func->params_.push_back(ca->var_);
    func->param_directions_.push_back(ParamDirection::Out);
  }
  std::vector<StmtPtr> kept;
  kept.reserve(seq->stmts_.size());
  for (const StmtPtr& s : seq->stmts_) {
    if (create_set.count(s.get()) == 0) kept.push_back(s);
  }
  func->body_ = SeqStmts::Flatten(std::move(kept), seq->span_);

  LOG_INFO << "AutoFuse[" << func->name_ << "]: wired " << creates.size()
           << " return(s) -> appended Out param(s) (device/harness binds outputs by param)";
}

ProgramPtr AutoFuseTransform(const ProgramPtr& prog) {
  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> new_functions;
  bool any_change = false;

  // Function names referenced by a Call/Submit anywhere in the program. A marked
  // function that is CALLED must keep its signature (MaybeLiftReturnToOutParam
  // skips it), so appending an Out param cannot break a callsite.
  std::unordered_set<std::string> called_funcs;
  {
    class CallCollector : public IRVisitor {
     public:
      std::unordered_set<std::string>* out = nullptr;
      const Program* prog = nullptr;

     protected:
      void VisitExpr_(const CallPtr& c) override {
        if (c->op_ && prog->GetFunction(c->op_->name_)) out->insert(c->op_->name_);
        IRVisitor::VisitExpr_(c);
      }
      void VisitExpr_(const SubmitPtr& s) override {
        if (s->op_ && prog->GetFunction(s->op_->name_)) out->insert(s->op_->name_);
        IRVisitor::VisitExpr_(s);
      }
    } cc;
    cc.out = &called_funcs;
    cc.prog = prog.get();
    for (const auto& [gv, fn] : prog->functions_) {
      if (fn->body_) cc.VisitStmt(fn->body_);
    }
  }
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
    // Empty problem (no compute ops) or out-of-scope (non-tensor output / dynamic shape)
    // -> leave the function untouched for legacy lowering.
    if (builder.declined() || builder.problem.ops.empty()) {
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
               << ot.width << "x" << ot.height << "]";
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
      const ::CostResult& cost = sol.step_cost(s);
      LOG_INFO << "  group[" << s << "] ops={" << members << "}  tile=" << step.config.w << "x"
               << step.config.h << (step.config.k ? ("x" + std::to_string(step.config.k)) : std::string())
               << "  split=" << cost.parallel_split << " cores=" << cost.cores_used
               << "  latency=" << sol.step_latency(s);
    }

    if (const char* dump_dir = std::getenv("PYPTO_AUTOFUSE_DUMP")) {
      const std::string base = std::string(dump_dir) + "/" + func->name_;
      DumpProblemJson(builder.problem, base + ".dag.json");
      DumpSolutionJson(sol, base + ".sol.json");
    }

    // Tier-B (A2 homogeneity): the generic emitter assumes each group is single-engine.
    // The base solver enforces that (allow_mixed off, ascend910b_cost.cpp:241), so a MIXED
    // cube+vector step can only appear under a PYPTO_FUSE_CUBE_VECTOR build. The emitter
    // would then SPLIT it across dispatch paths (matmul standalone, vector standalone) with
    // the intermediate through DDR — correct output, but NOT the on-chip-fused kernel the
    // solver costed. Surface it (S1 move-insertion is the real fix, not relaxing A2). Only
    // checked under the flag; legacy already splits mixed groups as existing behavior.
    if (GenericEmitEnabled()) {
      for (size_t s = 0; s < sol.num_steps(); ++s) {
        bool has_cube = false, has_vector = false;
        const auto& gops = sol.step(s).subgraph.ops();
        for (size_t op_idx : gops) {
          const ::OpType t = builder.problem.ops[op_idx].type;
          if (t == ::OpType::MatMul) has_cube = true;
          else if (t != ::OpType::Opaque) has_vector = true;  // Pointwise / Reduction -> vector
        }
        if (has_cube && has_vector)
          GenericDeclineB("mixed cube+vector group " + std::to_string(s) +
                              " (cross-engine, A2/S1) — realized as split kernels, not fused",
                          builder.op_stmts[gops.front()]->span_);
      }
    }

    // Emit: map each solver op back to its source statement, wrap each fused
    // group in an InCoreScopeStmt, and realize the chosen tile (step.config) for
    // matmuls — the output `[w,h]` tiling + the per-tile k-pipeline.
    std::unordered_map<const Stmt*, size_t> stmt_group;
    std::unordered_map<const Stmt*, SolverTile> stmt_tile;  // group's [w,h,k] tile, for matmul tiling
    std::unordered_map<const Stmt*, size_t> stmt_exec;      // solver's per-group pebbling order
    for (size_t s = 0; s < sol.num_steps(); ++s) {
      const ::TileConfig& cfg = sol.step(s).config;
      const SolverTile tile{cfg.w, cfg.h, cfg.k, sol.step_cost(s).parallel_split,
                            cfg.parts_m, cfg.parts_n};
      for (size_t op_idx : sol.step(s).subgraph.ops()) {
        const Stmt* stmt = builder.op_stmts[op_idx];
        stmt_group[stmt] = s;
        stmt_tile[stmt] = tile;
      }
      // The solver's execution_order() is the depth-first pebbling order it costed the
      // working-set peak along — the order the emit MUST replay to stay within UB.
      const std::vector<size_t>& exec = sol.step(s).subgraph.execution_order();
      for (size_t pos = 0; pos < exec.size(); ++pos) {
        stmt_exec[builder.op_stmts[exec[pos]]] = pos;
      }
    }
    auto new_func = MutableCopy(func);
    new_func->body_ = EmitFusedScopes(func->body_, stmt_group, stmt_tile, stmt_exec);
    // Wire the return-based fused function to a named Out param so orchestration
    // codegen emits an add_output write-back (device / ST harness bind output by
    // param position, not by return value). No-op for functions that already
    // have an Out param or are called internally.
    MaybeLiftReturnToOutParam(new_func, called_funcs);
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
