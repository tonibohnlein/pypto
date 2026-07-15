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
// The extractor builds PTO Fusebox's op+tensor DAG (`Problem`) from an
// `auto_fuse`-marked function by reusing PyPTO's own dependency analysis
// (`BuildStmtDependencyGraph`), which is Out/InOut/SSA-correct. This handles
// both forms uniformly:
//   * a flat tensor-level function (each AssignStmt is a tensor op), and
//   * an orchestration kernel-call DAG (`c_v1 = self.kernel_add(a, b, c)`),
//     where `tensor.create` allocations and Out-buffer args are skipped.
// The DAG is handed to the linked PTO Fusebox solver (`3rdparty/pto-fusebox`) to choose a
// memory-reuse fusion partition. v0 computes + logs (and optionally dumps) the
// grouping; the IR rewrite (emit InCoreScopeStmt) is the next increment.

#include <algorithm>
#include <any>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/soc.h"
#include "pypto/core/common.h"
#include "pypto/core/dtype.h"
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
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/attrs.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "pypto/ir/transforms/utils/stmt_dependency_analysis.h"
#include "pypto/ir/type.h"

// PTO Fusebox graph scheduler (3rdparty/pto-fusebox), linked as `solver_lib`.
#include "core/dag.h"
#include "core/flat_set.h"
#include "core/l0_matmul_plan.h"
#include "core/subgraph.h"
#include "core/types.h"
#include "partition/partition.h"
#include "pipeline/solver.h"
#include "solution/solution.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

// Hardware parameters. v0 hardcodes the Ascend 910B machine model (mirrors
// `set_910b` in 3rdparty/pto-fusebox/test/ascend_910b_test.cpp); the solver derives
// per-op compute from tile geometry (the grounded pto-isa fractal/vector model).
// TODO(cost-model): read these from BackendHandler instead of hardcoding 910B.
constexpr int64_t kFastMemoryCapacity = 1LL << 30;  // single-pool capacity hint
constexpr int kNumCubeCores = 24;               // AIC cores (matmul)
constexpr int kNumVectorCores = 48;             // AIV cores (pointwise / reduction)
constexpr int64_t kL1Capacity = 512LL * 1024;     // per-cube L1/Mat operand pool
constexpr int64_t kCubeCapacity = 128LL * 1024;   // per-cube L0c accumulator
constexpr int64_t kVecCapacity = 192LL * 1024;    // per-vector UB
constexpr int64_t kCubeComputeCost = 1;         // grounded per-repeat multiplier (cyc applies fp32 2x)
constexpr int64_t kKernelFillCost = 10000;      // per-kernel pipeline fill (cycles)
// Per-TASK host launch overhead, in the MODEL's cost-cycle scale (C3). The device grounded the
// launch term at ~0.2 us/task (compute-flat pointwise control). It must be added at the model's
// scale, NOT wall us: model cost under-represents wall by the calibration factor (~6.5x — e.g.
// rms[256,512] model 13011 cyc ~ 7 us nominal vs ~46 us wall), so 0.2 us_wall ~ 0.2us/6.5 ~ 57
// cycles. Empirically the window that ranks the three device-swept sizes correctly is (29, 115):
// below ~29 it can't flip rms[256,512] off its device-slowest argmin; above ~115 it over-corrects
// rms[512,1024] (picks fewer-but-slower tiles, overriding the makespan's parallelism preference).
// 64 sits mid-window and matches the calibrated 0.2 us. Verified: rms[256,512]->h=32 (device-fastest,
// was h=6/slowest), rms[512,1024]->h=32, pointwise[4096,64]->near-flat; solver suite unchanged.
// Refinable with a tighter op-sim-vs-wall clock-anchored calibration.
constexpr int64_t kPerTaskOverheadCycles = 64;

// Grounded pto-isa machine model (Ascend 910B / A2A3). Costs are in CORE CYCLES;
// bandwidths are GiB/s per direction (pto-isa arch_config.hpp). See the solver's
// types.h Problem::cube_freq_hz block for the exact formulas.
constexpr double kCubeFreqHz = 1.85e9;          // core clock (A2A3)
constexpr double kBwGmL1   = 135.0;             // GM->L1 operand reload
constexpr double kBwL0cGm  = 70.0;              // L0C->GM output store
constexpr double kBwL0cL1 = 128.0;              // L0C->L1 on-chip FIXPIPE drain
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
// Grounded vector (AIV) cost: per op = slope*repeat + head+tail (once per chain), repeat =
// ceil(elems / (vec_reg_bytes/dtype_bytes)). pto-isa A2A3 cce_costmodel_vector_compute.hpp,
// 910B3-calibrated (`标定`, dav-2201, R^2~1.0): 256-byte vreg (64 fp32 / 128 fp16 per repeat);
// per-op fixed (head+tail) ~24 (vadd) to ~31 (vexp); slope 2 for vadd/vsub/vmul/vexp, 4 for vdiv,
// 1 for vrsqrt/vrelu/vmuls. Per-op slope overrides are set per Op via VecOpSlope; the +16
// count-mode floor for unaligned width lives in the solver's VecOpCompute. The head/tail SPLIT
// is an unmeasured assumption upstream (only the ~32 SUM is used, charged once per chain).
constexpr int64_t kVecRegBytes = 256;           // vector register size (bytes)
constexpr double kVecOpHead = 14.0;             // per-op pipeline startup (head+tail sum is what matters)
constexpr double kVecOpTail = 18.0;             // per-op drain
constexpr double kVecSlopePw = 2.0;             // default elementwise cycles/repeat (vadd/vmul/vexp)
constexpr double kVecSlopeReduce = 14.0;        // DEPRECATED: solver uses the reduction TREE, not this

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
  // Output-buffer allocation, skipped by IsComputeOp. Exact IsOp, not a name substring: the old
  // find("create")/find("alloc") also matched an unrelated op named e.g. `my_alloc_scale`, silently
  // dropping its compute op from the solver graph (external-review finding). `tensor.create` is the
  // only allocation in the tensor-level auto_fuse DAG (`tensor.full` is a constant-fill = compute).
  return IsOp(call, "tensor.create");
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

// Hoist an inline compute expression in a ReturnStmt into a named binding so EVERY compute op is
// visible to the solver-graph builder (ProblemBuilder registers only `var = <call>` AssignStmts).
// The DSL allows `return pl.mul(a, b)` — an inline Call with no SSA name. That op never enters the
// solver graph, so the partitioner treats its operands as group-INTERNAL intermediates (not live-
// outs) and the emit drops them, leaving the raw inline return referencing an unexposed var. The
// bite is `return pl.mul(xc, iv)` where `xc` is a fused-group intermediate (layernorm/softmax
// written with a direct return): `xc` is buried in a group, unexposed, and the return dangles
// (BUG-LN-2). Rewrite `return <call>` to `_ret = <call>; return _ret`. Applied to the marked body
// BEFORE both ProblemBuilder::Build and EmitFusedScopes so the two walks stay consistent — and it
// gives the partitioner the true graph (a strictly better, cheaper partition). Returns nullopt when
// no return value is an inline compute call (a bare `return var` is already named). At this stage
// the body is a flat DAG (no control flow), so the ReturnStmt lives in the top-level SeqStmts.
std::optional<StmtPtr> HoistInlineReturnComputeExprs(const StmtPtr& body) {
  std::vector<StmtPtr> stmts;
  if (auto seq = As<SeqStmts>(body)) {
    stmts = seq->stmts_;
  } else {
    stmts.push_back(body);
  }
  std::vector<StmtPtr> out;
  out.reserve(stmts.size() + 2);
  bool changed = false;
  size_t hoist_idx = 0;
  for (const StmtPtr& s : stmts) {
    auto ret = As<ReturnStmt>(s);
    if (ret == nullptr) {
      out.push_back(s);
      continue;
    }
    std::vector<ExprPtr> new_vals;
    new_vals.reserve(ret->value_.size());
    for (const ExprPtr& v : ret->value_) {
      auto call = As<Call>(v);
      if (call != nullptr && !IsAllocCall(call)) {
        auto rv = std::make_shared<Var>("autofuse_ret" + std::to_string(hoist_idx++), v->GetType(),
                                        s->span_);
        out.push_back(std::make_shared<AssignStmt>(rv, v, s->span_));
        new_vals.push_back(ExprPtr(rv));
        changed = true;
      } else {
        new_vals.push_back(v);
      }
    }
    out.push_back(std::make_shared<ReturnStmt>(std::move(new_vals), s->span_));
  }
  if (!changed) {
    return std::nullopt;
  }
  return SeqStmts::Flatten(std::move(out), body->span_);
}

// One adapter-owned capability descriptor feeds solver admission and generic
// emission. Cost class alone is insufficient: product and arg reductions need
// different streaming state, while tensor.full has no strip-rewritable source
// operand. Until those algorithms exist they are explicit barriers, never
// accidental Pointwise fallbacks.
struct VectorOpDescriptor {
  VectorOpDescriptor(::OpType op_type, ::VectorOpCapability op_capability)
      : type(op_type), capability(op_capability) {}

  ::OpType type;
  ::VectorOpCapability capability;
};

VectorOpDescriptor DescribeVectorOp(const CallPtr& call) {
  const std::string& n = call->op_->name_;
  auto has = [&](const char* s) { return n.find(s) != std::string::npos; };
  auto any = [&](std::initializer_list<const char*> names) {
    for (const char* name : names) {
      // Capability admission deliberately includes retired and backend-specific
      // names so they remain explicit barriers if encountered in serialized IR.
      // Do not look those names up in OpRegistry: current main rightfully throws
      // for an unregistered legacy op (for example tensor.as_layout).
      if (n == name) return true;
    }
    return false;
  };
  if (has("matmul") || has("gemv")) {
    return VectorOpDescriptor(::OpType::MatMul, ::VectorOpCapability::Unsupported);
  }
  if (any({"tensor.row_sum", "tensor.col_sum"})) {
    return VectorOpDescriptor(::OpType::Reduction, ::VectorOpCapability::ReductionSum);
  }
  if (any({"tensor.row_max", "tensor.col_max"})) {
    return VectorOpDescriptor(::OpType::Reduction, ::VectorOpCapability::ReductionMax);
  }
  if (any({"tensor.row_min", "tensor.col_min", "tensor.row_prod", "tensor.col_prod",
           "tensor.row_argmax", "tensor.row_argmin", "tensor.col_argmax",
           "tensor.col_argmin"})) {
    return VectorOpDescriptor(::OpType::Opaque, ::VectorOpCapability::Unsupported);
  }
  if (has("gather") || has("scatter") || has("sort") || has("transpose") || has("reshape") ||
      has("concat") || has("assemble") ||
      any({"tensor.full", "tensor.random", "tensor.ci", "tensor.create", "tensor.slice",
           "tensor.fillpad", "tensor.fillpad_expand", "tensor.set_validshape",
           "tensor.as_layout", "tensor.row_expand", "tensor.col_expand",
           "tensor.expands", "tensor.expand_clone"})) {
    return VectorOpDescriptor(::OpType::Opaque, ::VectorOpCapability::Unsupported);
  }
  if (any({"tensor.add", "tensor.adds", "tensor.sub", "tensor.subs", "tensor.mul",
           "tensor.muls", "tensor.div", "tensor.divs", "tensor.fmod", "tensor.fmods",
           "tensor.maximum", "tensor.minimum", "tensor.cmp", "tensor.part_add",
           "tensor.part_mul", "tensor.part_max", "tensor.part_min", "tensor.abs",
           "tensor.cast", "tensor.cos", "tensor.exp", "tensor.log", "tensor.neg",
           "tensor.recip", "tensor.rsqrt", "tensor.sin", "tensor.sqrt",
           "tensor.row_expand_add", "tensor.row_expand_sub", "tensor.row_expand_mul",
           "tensor.row_expand_div", "tensor.row_expand_max", "tensor.row_expand_min",
           "tensor.row_expand_expdif", "tensor.col_expand_add", "tensor.col_expand_sub",
           "tensor.col_expand_mul", "tensor.col_expand_div", "tensor.col_expand_max",
           "tensor.col_expand_min", "tensor.col_expand_expdif"})) {
    return VectorOpDescriptor(::OpType::Pointwise, ::VectorOpCapability::Elementwise);
  }
  return VectorOpDescriptor(::OpType::Opaque, ::VectorOpCapability::Unsupported);
}

::OpType ClassifyOp(const CallPtr& call) {
  return DescribeVectorOp(call).type;
}

// Per-op VECTOR compute slope (cycles per SIMD repeat) when it differs from the elementwise
// default (~2). pto-isa vec_tile_study measured: most pointwise ops are slope 2, but the div
// family (`tensor.div` / `row_expand_div` / `col_expand_div` -> `vdiv`) is 4 and `tensor.rsqrt`
// (-> `vrsqrt`) is 1. Returns 0.0 = "use the group default vec_slope_pw". This is a cost
// heuristic keyed on the op FAMILY (not a correctness branch), so an unmatched op safely keeps
// the default; div-heavy kernels (softmax, RMS/LayerNorm) are the ones this de-underprices.
double VecOpSlope(const CallPtr& call) {
  const std::string& n = call->op_->name_;
  auto ends = [&](const char* s) {
    const size_t l = std::char_traits<char>::length(s);
    return n.size() >= l && n.compare(n.size() - l, l, s) == 0;
  };
  if (ends("div")) return 4.0;      // vdiv
  if (ends(".rsqrt")) return 1.0;   // vrsqrt (also vrelu/vmuls family, but rsqrt is the one in norms)
  return 0.0;                        // -> vec_slope_pw
}

// Per-op VECTOR fixed (head+tail) cycles, device-calibrated (pto-isa cce_costmodel_vector_compute):
// vadd/vsub/vmax/vmin 24, vmul 25, vexp/vln 31, vdiv 30, vrsqrt ~24. Charged once per chain (the
// stream-start op). Returns 0.0 = "use the group default `vec_op_head+vec_op_tail` (~32)". A cost
// heuristic keyed on the op family; unmatched ops keep the default.
double VecOpFixed(const CallPtr& call) {
  const std::string& n = call->op_->name_;
  auto ends = [&](const char* s) {
    const size_t l = std::char_traits<char>::length(s);
    return n.size() >= l && n.compare(n.size() - l, l, s) == 0;
  };
  if (ends("div")) return 30.0;                          // vdiv
  if (ends(".exp") || ends(".ln")) return 31.0;          // vexp / vln
  if (ends(".mul") || ends(".muls")) return 25.0;        // vmul / vmuls
  if (ends(".rsqrt") || ends(".sqrt")) return 24.0;      // vrsqrt / vsqrt
  if (ends(".add") || ends(".adds") || ends(".sub") || ends(".subs") || ends(".neg") ||
      ends(".max") || ends(".min") || ends(".maximum") || ends(".minimum")) {
    return 24.0;                                          // vadd / vsub / vmax / vmin family
  }
  return 0.0;                                             // -> vec_op_head + vec_op_tail
}

// Attach the exact pto-isa primitive and tile geometry that tensor-to-tile
// lowering will emit.  This classification happens once in the adapter after
// operand/output tensor ids are known; the solver then replays the descriptor
// at each VectorStreamPlan strip/chunk without inspecting PyPTO op names.
std::pair<::VectorPrimitiveFamily, ::VectorOpGeometry> ClassifyVectorOpSemantics(
    const CallPtr& call, const ::Op& op, const ::Problem& problem) {
  using Family = ::VectorPrimitiveFamily;
  using Geometry = ::VectorOpGeometry;

  if (IsOp(call, "tensor.row_sum")) return {Family::RowSum, Geometry::Flat};
  if (IsOp(call, "tensor.row_max")) {
    return {Family::RowExtrema, Geometry::Flat};
  }
  if (IsOp(call, "tensor.col_sum")) return {Family::ColSum, Geometry::Flat};
  if (IsOp(call, "tensor.col_max")) {
    return {Family::ColExtrema, Geometry::Flat};
  }

  Family family = Family::Generic;
  if (IsOp(call, "tensor.add") || IsOp(call, "tensor.sub") ||
      IsOp(call, "tensor.part_add") || IsOp(call, "tensor.part_max") ||
      IsOp(call, "tensor.part_min") ||
      IsOp(call, "tensor.row_expand_add") || IsOp(call, "tensor.row_expand_sub") ||
      IsOp(call, "tensor.row_expand_max") || IsOp(call, "tensor.row_expand_min") ||
      IsOp(call, "tensor.col_expand_add") || IsOp(call, "tensor.col_expand_sub") ||
      IsOp(call, "tensor.col_expand_max") || IsOp(call, "tensor.col_expand_min")) {
    family = Family::Add;
  } else if (IsOp(call, "tensor.mul") || IsOp(call, "tensor.part_mul") ||
             IsOp(call, "tensor.row_expand_mul") ||
             IsOp(call, "tensor.col_expand_mul")) {
    family = Family::Mul;
  } else if (IsOp(call, "tensor.div") || IsOp(call, "tensor.row_expand_div") ||
             IsOp(call, "tensor.col_expand_div")) {
    family = Family::Div;
  } else if (IsOp(call, "tensor.exp")) {
    family = Family::Exp;
  } else if (IsOp(call, "tensor.log")) {
    family = Family::Log;
  } else if (IsOp(call, "tensor.abs")) {
    family = Family::Abs;
  } else if (IsOp(call, "tensor.sqrt")) {
    family = Family::Sqrt;
  } else if (IsOp(call, "tensor.rsqrt")) {
    family = call->GetKwarg<bool>("high_precision", false) ? Family::Generic
                                                           : Family::Rsqrt;
  } else if (IsOp(call, "tensor.adds") || IsOp(call, "tensor.subs")) {
    family = Family::ScalarAdd;
  } else if (IsOp(call, "tensor.muls") || IsOp(call, "tensor.neg")) {
    family = Family::ScalarMul;
  } else if (IsOp(call, "tensor.maximum") || IsOp(call, "tensor.minimum")) {
    const bool scalar_rhs = call->args_.size() == 2 &&
                            As<ScalarType>(call->args_[1]->GetType()) != nullptr;
    if (!scalar_rhs) {
      family = Family::Add;  // tensor vmax/vmin share vadd's grounded class
    } else {
      family = IsOp(call, "tensor.maximum") ? Family::ScalarMax
                                             : Family::ScalarMin;
    }
  }
  if (family == Family::Generic) {
    return {Family::Generic, Geometry::Generic};
  }

  if (IsOp(call, "tensor.row_expand_add") || IsOp(call, "tensor.row_expand_sub") ||
      IsOp(call, "tensor.row_expand_mul") || IsOp(call, "tensor.row_expand_div") ||
      IsOp(call, "tensor.row_expand_max") || IsOp(call, "tensor.row_expand_min")) {
    return {family, Geometry::RowExpand};
  }
  if (IsOp(call, "tensor.col_expand_add") || IsOp(call, "tensor.col_expand_sub") ||
      IsOp(call, "tensor.col_expand_mul") || IsOp(call, "tensor.col_expand_div") ||
      IsOp(call, "tensor.col_expand_max") || IsOp(call, "tensor.col_expand_min")) {
    return {family, Geometry::ColExpand};
  }

  // Generic tensor.add/sub/mul/div are converted to a row/column expansion
  // when exactly one operand has the corresponding size-one axis.  A [1,1]
  // tensor broadcast is ambiguous, so retain the legacy descriptor instead of
  // claiming a particular lowering geometry.
  bool row_expand = false;
  bool col_expand = false;
  if (!op.outputs.empty()) {
    const ::Tensor& output = problem.tensors[op.output()];
    for (size_t input_id : op.inputs) {
      const ::Tensor& input = problem.tensors[input_id];
      row_expand |= input.width == 1 && output.width > 1 && input.height == output.height;
      col_expand |= input.height == 1 && output.height > 1 && input.width == output.width;
    }
  }
  if (row_expand != col_expand) {
    return {family, row_expand ? Geometry::RowExpand : Geometry::ColExpand};
  }
  if (row_expand) {
    return {Family::Generic, Geometry::Generic};
  }
  return {family, Geometry::Flat};
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

// Defined below (~the generic-emit dispatch); forward-declared so ProblemBuilder can gate the
// C3 per-task overhead on it (charge it only when the streaming emit that realizes fewer-tile
// plans is active).
static bool GenericEmitEnabled();

// Mixed cube/vector fusion is a separate bake gate.  It requires the generic
// tensor scheduler but remains opt-in until the C->V FIFO path is closed on
// silicon.
static bool MixedEmitEnabled();

// Exact nested L0 costing is deliberately independent of generic GM/L1
// emission. The analytic mode remains the default; exact mode is an opt-in
// compile-time/experimentation policy until its silicon benefit justifies the
// additional candidate-evaluation cost.
static bool ExactL0CostEnabled();

// Defined below; forward-declared so ProblemBuilder registers exact P4 algorithm descriptors only
// when the corresponding online emitter is active.
static bool P4Enabled(::P4PatternKind kind);

// Defined below (~line 788); forward-declared so ProblemBuilder's A1 gate can read a 2D tensor shape.
std::pair<int64_t, int64_t> Static2DShape(const TypePtr& type);

// One semantic analysis feeds both sides of the P4 fidelity contract. The solver receives the exact
// matched op set through Problem::p4_patterns; the emitter receives the corresponding handles below.
// Neither side is allowed to rediscover a looser "looks like softmax/layernorm" shape.
struct P4Match {
  ::P4PatternKind kind = ::P4PatternKind::None;
  std::vector<AssignStmtPtr> ops;
  AssignStmtPtr sink;
  AssignStmtPtr max_stmt;
  AssignStmtPtr sum_stmt;
  std::vector<AssignStmtPtr> layernorm_sums;  // {sum(x), sum(x*x)}
  ExprPtr x_input;
  VarPtr user_mean;
  VarPtr user_var;
};

// Match only the algorithms the current P4 emit actually implements:
//
//   softmax:  m=max(x); sh=x-m; e=exp(sh); s=sum(e); out=e/s
//   layernorm: sx=sum(x); sxx=sum(x*x); mean=sx/N; var=sxx/N-mean*mean;
//              inv=rsqrt(var+eps); out=(x-mean)*inv
//
// This is deliberately exact. A temperature-scaled softmax, weighted second moment, affine tail, or
// any other near miss is a different algorithm and must be cut until it has its own proven descriptor.
// The scan is O(N): every candidate sink follows only a fixed-depth canonical chain.
std::vector<P4Match> AnalyzeP4Patterns(const std::vector<StmtPtr>& stmts) {
  std::vector<AssignStmtPtr> ops;
  std::unordered_map<const Var*, AssignStmtPtr> defmap;
  for (const StmtPtr& stmt : stmts) {
    CallPtr call;
    if (!IsComputeOp(stmt, &call)) continue;
    auto assign = As<AssignStmt>(stmt);
    ops.push_back(assign);
    defmap.emplace(assign->var_.get(), assign);
  }
  std::unordered_map<const Var*, std::vector<const Stmt*>> consumers;
  for (const AssignStmtPtr& op : ops) {
    for (const ExprPtr& arg : As<Call>(op->value_)->args_) {
      auto var = AsVarLike(arg);
      if (var != nullptr && defmap.count(var.get()) != 0) consumers[var.get()].push_back(op.get());
    }
  }
  for (const StmtPtr& stmt : stmts) {
    auto ret = As<ReturnStmt>(stmt);
    if (ret == nullptr) continue;
    for (const ExprPtr& value : ret->value_) {
      auto var = AsVarLike(value);
      if (var != nullptr && defmap.count(var.get()) != 0) {
        consumers[var.get()].push_back(ret.get());
      }
    }
  }

  auto def_of = [&](const ExprPtr& expr) -> AssignStmtPtr {
    auto var = AsVarLike(expr);
    if (var == nullptr) return nullptr;
    auto it = defmap.find(var.get());
    return it == defmap.end() ? nullptr : it->second;
  };
  auto call_of = [](const AssignStmtPtr& stmt) -> CallPtr {
    return stmt == nullptr ? nullptr : As<Call>(stmt->value_);
  };
  // A vector-only P4 cone may consume either a graph boundary or the result of
  // the immediately preceding cube stage.  The online algorithm is identical
  // in both cases; the latter is the mixed QK -> softmax building block.
  auto is_boundary_or_matmul_source = [&](const ExprPtr& expr) {
    const AssignStmtPtr producer = def_of(expr);
    return producer == nullptr || ClassifyOp(call_of(producer)) == ::OpType::MatMul;
  };
  auto same_var = [](const ExprPtr& lhs, const ExprPtr& rhs) -> bool {
    auto lv = AsVarLike(lhs);
    auto rv = AsVarLike(rhs);
    return lv != nullptr && rv != nullptr && lv.get() == rv.get();
  };
  auto is_scalar_const = [](const ExprPtr& expr) -> bool {
    return As<ConstFloat>(expr) != nullptr || As<ConstInt>(expr) != nullptr;
  };
  auto scaled_from = [&](const AssignStmtPtr& stmt, const ExprPtr& input, double scale) -> bool {
    auto call = call_of(stmt);
    if (call == nullptr || call->args_.size() != 2) return false;
    if (IsOp(call, "tensor.muls")) {
      return same_var(call->args_[0], input) && IsConstValue(call->args_[1], scale);
    }
    if (!IsOp(call, "tensor.mul")) return false;
    return (same_var(call->args_[0], input) && IsConstValue(call->args_[1], scale)) ||
           (same_var(call->args_[1], input) && IsConstValue(call->args_[0], scale));
  };

  auto match_softmax = [&](const AssignStmtPtr& sink) -> std::optional<P4Match> {
    auto div = call_of(sink);
    if (div == nullptr || div->args_.size() != 2 ||
        (!IsOp(div, "tensor.row_expand_div") && !IsOp(div, "tensor.div"))) {
      return std::nullopt;
    }
    auto exp_stmt = def_of(div->args_[0]);
    auto sum_stmt = def_of(div->args_[1]);
    auto exp = call_of(exp_stmt);
    auto sum = call_of(sum_stmt);
    if (exp == nullptr || sum == nullptr || exp->args_.size() != 1 || sum->args_.size() != 1 ||
        !IsOp(exp, "tensor.exp") || !IsOp(sum, "tensor.row_sum") ||
        !same_var(sum->args_[0], ExprPtr(exp_stmt->var_))) {
      return std::nullopt;
    }
    auto sub_stmt = def_of(exp->args_[0]);
    auto sub = call_of(sub_stmt);
    if (sub == nullptr || sub->args_.size() != 2 ||
        (!IsOp(sub, "tensor.row_expand_sub") && !IsOp(sub, "tensor.sub"))) {
      return std::nullopt;
    }
    auto max_stmt = def_of(sub->args_[1]);
    auto max = call_of(max_stmt);
    if (max == nullptr || max->args_.size() != 1 || !IsOp(max, "tensor.row_max") ||
        !same_var(sub->args_[0], max->args_[0]) || !is_boundary_or_matmul_source(max->args_[0])) {
      return std::nullopt;  // direct boundary x, or a direct MatMul-produced mixed-stage x
    }
    const auto [xM, xN] = Static2DShape(max->args_[0]->GetType());
    const auto [oM, oN] = Static2DShape(sink->var_->GetType());
    const auto [mM, mN] = Static2DShape(max_stmt->var_->GetType());
    const auto [sM, sN] = Static2DShape(sum_stmt->var_->GetType());
    if (xM <= 0 || xN <= 1 || oM != xM || oN != xN || mM != xM || mN != 1 || sM != xM || sN != 1) {
      return std::nullopt;
    }
    P4Match match;
    match.kind = ::P4PatternKind::SoftmaxFlash;
    match.ops = {max_stmt, sub_stmt, exp_stmt, sum_stmt, sink};
    match.sink = sink;
    match.max_stmt = max_stmt;
    match.sum_stmt = sum_stmt;
    match.x_input = max->args_[0];
    return match;
  };

  auto match_layernorm = [&](const AssignStmtPtr& sink) -> std::optional<P4Match> {
    auto mul_out = call_of(sink);
    if (mul_out == nullptr || mul_out->args_.size() != 2 ||
        (!IsOp(mul_out, "tensor.row_expand_mul") && !IsOp(mul_out, "tensor.mul"))) {
      return std::nullopt;
    }

    ExprPtr centered_expr = mul_out->args_[0];
    ExprPtr inv_expr = mul_out->args_[1];
    auto centered_stmt = def_of(centered_expr);
    auto inv_stmt = def_of(inv_expr);
    auto centered = call_of(centered_stmt);
    auto inv = call_of(inv_stmt);
    // tensor.mul is commutative; accept the inverse/centered operands in either order.
    if (IsOp(mul_out, "tensor.mul") && (centered == nullptr || (!IsOp(centered, "tensor.row_expand_sub") &&
                                                                !IsOp(centered, "tensor.sub")))) {
      std::swap(centered_expr, inv_expr);
      std::swap(centered_stmt, inv_stmt);
      centered = call_of(centered_stmt);
      inv = call_of(inv_stmt);
    }
    if (centered == nullptr || centered->args_.size() != 2 ||
        (!IsOp(centered, "tensor.row_expand_sub") && !IsOp(centered, "tensor.sub")) || inv == nullptr ||
        inv->args_.size() != 1 || !IsOp(inv, "tensor.rsqrt")) {
      return std::nullopt;
    }

    auto mean_stmt = def_of(centered->args_[1]);
    if (mean_stmt == nullptr) {
      return std::nullopt;
    }
    ExprPtr var_expr = inv->args_[0];
    AssignStmtPtr eps_stmt;
    if (auto maybe_eps = def_of(var_expr)) {
      auto eps = call_of(maybe_eps);
      if (eps != nullptr && eps->args_.size() == 2 && (IsOp(eps, "tensor.adds") || IsOp(eps, "tensor.add"))) {
        if (is_scalar_const(eps->args_[1])) {
          var_expr = eps->args_[0];
          eps_stmt = maybe_eps;
        } else if (IsOp(eps, "tensor.add") && is_scalar_const(eps->args_[0])) {
          var_expr = eps->args_[1];
          eps_stmt = maybe_eps;
        }
      }
    }
    auto var_stmt = def_of(var_expr);
    auto var = call_of(var_stmt);
    if (var == nullptr || var->args_.size() != 2 || !IsOp(var, "tensor.sub")) {
      return std::nullopt;
    }
    auto msq_stmt = def_of(var->args_[0]);
    auto mean_sq_stmt = def_of(var->args_[1]);
    auto mean_sq = call_of(mean_sq_stmt);
    if (mean_sq == nullptr || mean_sq->args_.size() != 2 || !IsOp(mean_sq, "tensor.mul") ||
        !same_var(mean_sq->args_[0], ExprPtr(mean_stmt->var_)) ||
        !same_var(mean_sq->args_[1], ExprPtr(mean_stmt->var_))) {
      return std::nullopt;
    }

    const ExprPtr x = centered->args_[0];
    const auto [xM, xN] = Static2DShape(x->GetType());
    const auto [oM, oN] = Static2DShape(sink->var_->GetType());
    if (xM <= 0 || xN <= 1 || oM != xM || oN != xN || !is_boundary_or_matmul_source(x)) {
      return std::nullopt;  // both sums reduce a boundary or direct MatMul-produced x
    }
    const double inv_n = 1.0 / static_cast<double>(xN);

    auto sx_stmt = def_of(call_of(mean_stmt) != nullptr && !call_of(mean_stmt)->args_.empty()
                              ? call_of(mean_stmt)->args_[0]
                              : nullptr);
    auto sxx_stmt =
        def_of(call_of(msq_stmt) != nullptr && !call_of(msq_stmt)->args_.empty() ? call_of(msq_stmt)->args_[0]
                                                                                 : nullptr);
    if (!scaled_from(mean_stmt, sx_stmt != nullptr ? ExprPtr(sx_stmt->var_) : nullptr, inv_n) ||
        !scaled_from(msq_stmt, sxx_stmt != nullptr ? ExprPtr(sxx_stmt->var_) : nullptr, inv_n)) {
      return std::nullopt;
    }
    auto sx = call_of(sx_stmt);
    auto sxx = call_of(sxx_stmt);
    if (sx == nullptr || sxx == nullptr || sx->args_.size() != 1 || sxx->args_.size() != 1 ||
        !IsOp(sx, "tensor.row_sum") || !IsOp(sxx, "tensor.row_sum") || !same_var(sx->args_[0], x)) {
      return std::nullopt;
    }
    auto square_stmt = def_of(sxx->args_[0]);
    auto square = call_of(square_stmt);
    if (square == nullptr || square->args_.size() != 2 || !IsOp(square, "tensor.mul") ||
        !same_var(square->args_[0], x) || !same_var(square->args_[1], x)) {
      return std::nullopt;
    }
    const auto [meanM, meanN] = Static2DShape(mean_stmt->var_->GetType());
    const auto [varM, varN] = Static2DShape(var_stmt->var_->GetType());
    if (meanM != xM || meanN != 1 || varM != xM || varN != 1) {
      return std::nullopt;
    }

    std::vector<AssignStmtPtr> matched = {sx_stmt,  square_stmt,  sxx_stmt, mean_stmt,
                                          msq_stmt, mean_sq_stmt, var_stmt};
    if (eps_stmt != nullptr) matched.push_back(eps_stmt);
    matched.insert(matched.end(), {inv_stmt, centered_stmt, sink});
    P4Match match;
    match.kind = ::P4PatternKind::LayerNormWelford;
    match.ops = std::move(matched);
    match.sink = sink;
    match.layernorm_sums = {sx_stmt, sxx_stmt};
    match.x_input = x;
    match.user_mean = mean_stmt->var_;
    match.user_var = var_stmt->var_;
    return match;
  };

  // The online kernel has one live-out. If an internal statistic/cone value also feeds an op outside
  // the exact pattern, this candidate would be multi-sink even though its op set still matches.
  auto has_single_live_out = [&](const P4Match& match) -> bool {
    std::unordered_set<const Stmt*> members;
    for (const AssignStmtPtr& op : match.ops) members.insert(op.get());
    for (const AssignStmtPtr& op : match.ops) {
      if (op == match.sink) continue;
      auto it = consumers.find(op->var_.get());
      if (it == consumers.end()) continue;
      for (const Stmt* consumer : it->second) {
        if (members.count(consumer) == 0) {
          return false;
        }
      }
    }
    return true;
  };

  std::vector<P4Match> matches;
  for (const AssignStmtPtr& sink : ops) {
    if (auto softmax = match_softmax(sink)) {
      if (has_single_live_out(*softmax)) matches.push_back(std::move(*softmax));
      continue;
    }
    if (auto layernorm = match_layernorm(sink); layernorm && has_single_live_out(*layernorm)) {
      matches.push_back(std::move(*layernorm));
    }
  }
  return matches;
}

// Build the PTO Fusebox `Problem` (op+tensor DAG) from a function, reusing
// `BuildStmtDependencyGraph` for sound op-dependency edges.
class ProblemBuilder {
 public:
  ::Problem problem;
  std::vector<std::string> op_labels;     // per-op kernel/op name, for readable logging
  std::vector<const Stmt*> op_stmts;      // per-op source AssignStmt (op index -> Stmt), for the emit
  std::vector<P4Match> p4_matches;        // exact semantic matches shared by model + emit

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
    // C3 per-task launch overhead. GATED on the generic emit: it steers the solver toward FEWER,
    // larger tiles, which ONLY the generic emit can build (its stage-2 pipeline UB-streams a large
    // tile; the legacy TilePointwiseGroup materializes the whole tile and would overflow UB). Pricing
    // per-task overhead for the legacy path would pick tiles that path cannot realize (a §0 contract
    // violation — price what you build), so charge it only when the streaming emit is active.
    problem.per_task_overhead_cycles = GenericEmitEnabled() ? kPerTaskOverheadCycles : 0;
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
    // The cube scheduler and AutoTileMatmulL0 share one lower-level planner.
    // Only backend/memory-planner facts live here; each cube request supplies
    // its concrete M/N/K, operand widths, accumulator-read flag, and output
    // target when deriving its child L0MatmulPlan.
    {
      const auto* pctx = PassContext::Current();
      const auto* handler = pctx ? pctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
      INTERNAL_CHECK(handler)
          << "Internal error: BackendHandler is null while configuring AutoFuse L0 planner";
      auto& l0 = problem.l0_matmul_config;
      l0.l0a_bytes = handler->GetL0aCapacityBytes();
      l0.l0b_bytes = handler->GetL0bCapacityBytes();
      l0.l0c_bytes = handler->GetL0cCapacityBytes();
      l0.align_m = handler->GetL0FractalAlignment();
      l0.align_n = handler->GetL0FractalAlignment();
      l0.align_k = handler->GetL0FractalAlignment();
      l0.min_m = handler->GetMinL0TileDim();
      l0.min_n = handler->GetMinL0TileDim();
      l0.min_k = handler->GetMinL0TileDim();
      const auto cost_model = handler->GetL0CostModel();
      l0.bw_l0a = cost_model.bw_l0a;
      l0.bw_l0b = cost_model.bw_l0b;
      l0.bw_drain = cost_model.bw_drain;
      l0.bw_l0c_l1 = kBwL0cL1 * static_cast<double>(1ULL << 30) / kCubeFreqHz;
      l0.drain_fixed_cycles = cost_model.drain_fixed_cycles;
      l0.drain_row_cycles = cost_model.drain_row_cycles;
      l0.drain_penalty_cycles = cost_model.drain_penalty_cycles;
      l0.drain_c0_bytes = cost_model.drain_c0_bytes;
      l0.mad_head_cycles = cost_model.mad_head_cycles;
      l0.mad_k_fractal_bytes = cost_model.mad_k_fractal_bytes;
      l0.allow_a_stationary = true;
      l0.allow_b_stationary = true;
      l0.allow_double_buffer_c =
          pctx &&
          (pctx->GetMemoryPlanner() == MemoryPlanner::PtoAS ||
           (pctx->GetMemoryPlanner() == MemoryPlanner::PyPTO && pctx->GetEnablePyptoL0cDoubleBuffer()));
      l0.allow_padding = false;
      l0.allow_k_boundary = true;
    }
    problem.vec_reg_bytes = kVecRegBytes;
    // DMA-block granule (bytes) from the backend handler — keeps the cost model's tile-footprint
    // padding (vector_peak_ub) in lockstep with the emit's AlignUp tile allocation so a thin free
    // axis is not under-counted as UB-feasible (BUG-G1THRESH). Falls back to the field default (32).
    {
      const auto* pctx = PassContext::Current();
      const auto* h = pctx ? pctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
      if (h != nullptr) problem.vec_dma_align_bytes = h->GetVectorDmaAlignmentBytes();
    }
    problem.vec_op_head = kVecOpHead;
    problem.vec_op_tail = kVecOpTail;
    problem.vec_slope_pw = kVecSlopePw;
    problem.vec_slope_reduce = kVecSlopeReduce;
    // BUILDABLE mode. The analytic override stays false; exact P4 op sets discovered by the single
    // semantic analysis below are registered in problem.p4_patterns. Thus the cost model can admit
    // only the same complete algorithm descriptor the emitter will consume.
    problem.allow_model_ahead_multi_reduction_stream = false;
    // The current cross-core skew pass supports at most one cube<->vector
    // round trip. Keep deeper mixed pipelines analytic-only until their FIFO
    // schedule and emit are implemented.
    problem.allow_model_ahead_mixed_multi_roundtrip = false;
    problem.fuse_cube_vector = GenericEmitEnabled() && MixedEmitEnabled();
    problem.require_buildable_mixed = problem.fuse_cube_vector;
    // Multi-matmul CubeSchedulePlan emission currently requires one static
    // region shape per SPMD body. Keep analytic solver users unrestricted, but
    // do not let the compiler select a balanced non-uniform DAG grid that it
    // would have to materialize as standalone fallback kernels.
    problem.require_uniform_cube_dag_grid = GenericEmitEnabled();
    problem.use_hierarchical_cube_cost = ExactL0CostEnabled();

    // 1. In AND InOut params are graph-input tensors: both are READ by the body. An InOut param
    //    is also written, but its updated value is a SEPARATE SSA tensor produced by a compute op
    //    (which becomes a live-out on its own); the param itself carries the initial value = a
    //    boundary input. Omitting InOut here would leave an op that reads it with an incomplete
    //    input set (undercounting its DDR read, or making it appear input-less). Out params are
    //    write-only boundary buffers, never a graph input.
    for (size_t i = 0; i < func->params_.size(); ++i) {
      if (i >= func->param_directions_.size()) continue;
      const ParamDirection dir = func->param_directions_[i];
      if (dir == ParamDirection::In || dir == ParamDirection::InOut) {
        in_params_.insert(func->params_[i].get());
        // Only tensor-typed params are tiled tensors in the solver's model; a scalar In-param
        // (e.g. a broadcast scale) is an operand carried through the emit as-is, never a tracked
        // tensor -> skip it. Registering it would trip TensorId's tensor-type decline and
        // needlessly abandon a fusable function.
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

    // Function results are observable graph boundaries even when another op
    // consumes the same SSA value. Record them before subgraph construction so
    // costing, liveness, P4 escape checks, and emission share that fact.
    for (const StmtPtr& stmt : dep.stmts) {
      auto ret = As<ReturnStmt>(stmt);
      if (ret == nullptr) continue;
      for (const ExprPtr& value : ret->value_) {
        auto var = AsVarLike(value);
        if (var == nullptr) continue;
        auto it = tensor_index_.find(var.get());
        if (it != tensor_index_.end()) problem.required_outputs.insert(it->second);
      }
    }

    // 4. Second pass: emit ops. Inputs = predecessor-op outputs (from the
    //    dependency graph) + In-param args. Out-buffers/allocs fall out because
    //    they are never registered as tensors.
    bool has_matmul_call = false;
    bool has_vector_call = false;
    for (const auto& [unused_stmt, candidate] : ops) {
      (void)unused_stmt;
      if (DescribeVectorOp(candidate).type == ::OpType::MatMul) {
        has_matmul_call = true;
      } else {
        has_vector_call = true;
      }
    }
    const bool mixed_function = MixedEmitEnabled() && has_matmul_call && has_vector_call;
    for (const auto& entry : ops) {
      const Stmt* stmt = entry.first;
      const CallPtr& call = entry.second;
      ::Op sop;
      const VectorOpDescriptor descriptor = DescribeVectorOp(call);
      sop.type = descriptor.type;
      sop.vector_capability = descriptor.capability;
      if (sop.type == ::OpType::MatMul) {
        const bool default_semantics =
            !call->GetKwarg<bool>("a_trans", false) &&
            !call->GetKwarg<bool>("b_trans", false) &&
            !call->GetKwarg<bool>("c_matrix_nz", false);
        const auto lhs_type = call->args_.empty()
                                  ? nullptr
                                  : AsTensorTypeLike(call->args_[0]->GetType());
        const auto rhs_type = call->args_.size() < 2
                                  ? nullptr
                                  : AsTensorTypeLike(call->args_[1]->GetType());
        const bool same_operands = lhs_type && rhs_type &&
                                   lhs_type->dtype_ == rhs_type->dtype_;
        const bool pto_cube_dtype = same_operands &&
                                    (lhs_type->dtype_ == DataType::FP32 ||
                                     lhs_type->dtype_ == DataType::FP16 ||
                                     lhs_type->dtype_ == DataType::BF16 ||
                                     lhs_type->dtype_ == DataType::INT8);
        const bool compiler_mixed_dtype = same_operands &&
                                          (lhs_type->dtype_ == DataType::FP32 ||
                                           lhs_type->dtype_ == DataType::FP16 ||
                                           lhs_type->dtype_ == DataType::BF16);
        sop.mixed_emit_compatible = default_semantics && compiler_mixed_dtype;
        // The existing homogeneous tiler also assumes default orientation.
        // Its tile lowering also requires equal PTO-supported operand dtypes.
        // While mixed emission is explicitly requested, leave an unsafe source
        // function wholly untouched rather than rejecting the mixed group and
        // then crashing in the standalone fallback. INT8 is fallback-safe but
        // still not admitted to mixed v0 (no integer vector-family table yet).
        if (mixed_function && (!default_semantics || !pto_cube_dtype)) declined_ = true;
      }
      sop.vec_slope = VecOpSlope(call);  // vdiv=4 / vrsqrt=1 override the elementwise default
      sop.vec_fixed = VecOpFixed(call);  // per-op head+tail (vadd 24 / vexp 31 / vdiv 30 / ...)
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
      std::tie(sop.vector_primitive, sop.vector_geometry) =
          ClassifyVectorOpSemantics(call, sop, problem);
      if (mixed_function && sop.type == ::OpType::Pointwise &&
          sop.vector_capability == ::VectorOpCapability::Elementwise) {
        const ::DType output_dtype = problem.tensors[out].dtype;
        for (size_t input : sop.inputs) {
          if (problem.tensors[input].dtype != output_dtype) {
            // The emitted PTO primitive has no implicit tensor promotion. A
            // cut would reach the same tile op, so fail closed at the source.
            declined_ = true;
          }
        }
      }
      problem.ops.push_back(std::move(sop));
      op_labels.push_back(call->op_->name_);
      op_stmts.push_back(stmt);
    }

    if (GenericEmitEnabled()) {
      std::unordered_map<const Stmt*, size_t> op_index;
      for (size_t i = 0; i < op_stmts.size(); ++i) op_index.emplace(op_stmts[i], i);
      for (P4Match& match : AnalyzeP4Patterns(dep.stmts)) {
        if (!P4Enabled(match.kind)) continue;
        FlatSet<size_t> matched_ops;
        bool complete = true;
        for (const AssignStmtPtr& stmt : match.ops) {
          auto it = op_index.find(stmt.get());
          if (it == op_index.end()) {
            complete = false;
            break;
          }
          matched_ops.insert(it->second);
        }
        if (!complete || matched_ops.size() != match.ops.size()) continue;
        FlatSet<size_t> apply_substitutions;
        auto add_substitution = [&](const AssignStmtPtr& stmt) {
          if (stmt == nullptr) return;
          auto it = op_index.find(stmt.get());
          if (it != op_index.end()) apply_substitutions.insert(it->second);
        };
        if (match.kind == ::P4PatternKind::SoftmaxFlash) {
          add_substitution(match.max_stmt);
          add_substitution(match.sum_stmt);
        } else if (match.kind == ::P4PatternKind::LayerNormWelford) {
          for (const AssignStmtPtr& stmt : match.ops) {
            if (stmt->var_.get() == match.user_mean.get() || stmt->var_.get() == match.user_var.get()) {
              add_substitution(stmt);
            }
          }
        }
        ::P4Pattern pattern;
        pattern.kind = match.kind;
        pattern.ops = std::move(matched_ops);
        pattern.apply_substitutions = std::move(apply_substitutions);
        problem.p4_patterns.push_back(std::move(pattern));
        p4_matches.push_back(std::move(match));
      }
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
    DataType dtype = DataType::FP32;
    if (tt != nullptr) {
      dtype = tt->dtype_;
    }
    const size_t idx = problem.tensors.size();
    ::Tensor tensor;
    tensor.width = w;
    tensor.height = h;
    tensor.dtype = MapSolverDType(dtype);
    problem.tensors.push_back(tensor);
    tensor_index_[raw] = idx;
    return idx;
  }

  // Returns false (and leaves *w,*h at a safe placeholder) if any dim is dynamic/symbolic OR the
  // tensor is rank>2. A rank>=3 tensor read as its last two dims would UNDERCOUNT the solver's cost
  // by the product of the leading dims (a [B,M,N] tensor priced as [M,N]) AND never examine dim 0
  // (so a dynamic batch dim would pass) — the emit only handles 2D (Static2DShape hard-requires
  // rank 2), so a rank>=3 operand is out of scope; decline the whole function instead.
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
    if (shape.size() == 2) {
      *h = dim(0);
      *w = dim(1);
    } else if (shape.size() == 1) {
      *w = dim(0);
      *h = 1;
    } else if (shape.size() == 0) {
      *w = 1;
      *h = 1;
    } else {  // rank >= 3 -> out of scope (would undercount + miss a dynamic leading dim)
      *w = 1;
      *h = 1;
      ok = false;
    }
    return ok;
  }
};

// Dump the extracted DAG as a competition-format JSON instance (for
// visualization via 3rdparty/pto-fusebox/scripts/visualize.py). Hand-rolled JSON.
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
    const char* s = nullptr;
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
  const auto op_type_name = [](::OpType type) -> const char* {
    switch (type) {
      case ::OpType::MatMul: return "MatMul";
      case ::OpType::Pointwise: return "Pointwise";
      case ::OpType::Reduction: return "Reduction";
      case ::OpType::Opaque: return "Opaque";
    }
    return "Opaque";
  };
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "\"" << op_type_name(p.ops[i].type) << "\"";
  }
  const auto primitive_name = [](::VectorPrimitiveFamily family) -> const char* {
    switch (family) {
      case ::VectorPrimitiveFamily::Generic: return "generic";
      case ::VectorPrimitiveFamily::Add: return "add";
      case ::VectorPrimitiveFamily::Mul: return "mul";
      case ::VectorPrimitiveFamily::Div: return "div";
      case ::VectorPrimitiveFamily::Exp: return "exp";
      case ::VectorPrimitiveFamily::Log: return "log";
      case ::VectorPrimitiveFamily::Abs: return "abs";
      case ::VectorPrimitiveFamily::Sqrt: return "sqrt";
      case ::VectorPrimitiveFamily::Rsqrt: return "rsqrt";
      case ::VectorPrimitiveFamily::ScalarAdd: return "scalar_add";
      case ::VectorPrimitiveFamily::ScalarMul: return "scalar_mul";
      case ::VectorPrimitiveFamily::ScalarMax: return "scalar_max";
      case ::VectorPrimitiveFamily::ScalarMin: return "scalar_min";
      case ::VectorPrimitiveFamily::RowSum: return "row_sum";
      case ::VectorPrimitiveFamily::RowExtrema: return "row_extrema";
      case ::VectorPrimitiveFamily::ColSum: return "col_sum";
      case ::VectorPrimitiveFamily::ColExtrema: return "col_extrema";
      case ::VectorPrimitiveFamily::Reduction: return "reduction";
    }
    return "generic";
  };
  const auto geometry_name = [](::VectorOpGeometry geometry) -> const char* {
    switch (geometry) {
      case ::VectorOpGeometry::Generic: return "generic";
      case ::VectorOpGeometry::Flat: return "flat";
      case ::VectorOpGeometry::RowExpand: return "row_expand";
      case ::VectorOpGeometry::ColExpand: return "col_expand";
    }
    return "generic";
  };
  const auto capability_name = [](::VectorOpCapability capability) -> const char* {
    switch (capability) {
      case ::VectorOpCapability::Generic: return "generic";
      case ::VectorOpCapability::Elementwise: return "elementwise";
      case ::VectorOpCapability::ReductionSum: return "reduction_sum";
      case ::VectorOpCapability::ReductionMax: return "reduction_max";
      case ::VectorOpCapability::Unsupported: return "unsupported";
    }
    return "unsupported";
  };
  f << "],\n  \"vec_slopes\": [";
  for (size_t i = 0; i < no; ++i) f << (i ? "," : "") << p.ops[i].vec_slope;
  f << "],\n  \"vec_fixed_costs\": [";
  for (size_t i = 0; i < no; ++i) f << (i ? "," : "") << p.ops[i].vec_fixed;
  f << "],\n  \"vector_primitive_families\": [";
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "\"" << primitive_name(p.ops[i].vector_primitive) << "\"";
  }
  f << "],\n  \"vector_op_geometries\": [";
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "\"" << geometry_name(p.ops[i].vector_geometry) << "\"";
  }
  f << "],\n  \"vector_op_capabilities\": [";
  for (size_t i = 0; i < no; ++i) {
    f << (i ? "," : "") << "\"" << capability_name(p.ops[i].vector_capability) << "\"";
  }
  f << "],\n  \"required_outputs\": [";
  bool first_required = true;
  for (size_t tensor : p.required_outputs) {
    f << (first_required ? "" : ",") << tensor;
    first_required = false;
  }
  f << "],\n  \"fast_memory_capacity\": " << p.fast_memory_capacity;
  // 910B topology + grounded pto-isa machine model — emit so a dumped instance
  // re-loads (io.cpp) into the SAME grounded cost path the pass solved with.
  f << ",\n  \"num_cube_cores\": " << p.num_cube_cores << ",\n  \"num_vector_cores\": " << p.num_vector_cores
    << ",\n  \"fuse_cube_vector\": " << (p.fuse_cube_vector ? "true" : "false")
    << ",\n  \"require_buildable_mixed\": "
    << (p.require_buildable_mixed ? "true" : "false")
    << ",\n  \"allow_model_ahead_mixed_multi_roundtrip\": "
    << (p.allow_model_ahead_mixed_multi_roundtrip ? "true" : "false")
    << ",\n  \"cube_capacity\": " << p.cube_capacity << ",\n  \"vec_capacity\": " << p.vec_capacity
    << ",\n  \"l1_capacity\": " << p.l1_capacity << ",\n  \"cube_compute_cost\": " << p.cube_compute_cost
    << ",\n  \"kernel_fill_cost\": " << p.kernel_fill_cost
    << ",\n  \"per_task_overhead_cycles\": " << p.per_task_overhead_cycles
    << ",\n  \"cube_freq_hz\": " << p.cube_freq_hz << ",\n  \"bw_gm_l1\": " << p.bw_gm_l1
    << ",\n  \"bw_l0c_gm\": " << p.bw_l0c_gm << ",\n  \"bw_l1_l0a\": " << p.bw_l1_l0a
    << ",\n  \"bw_l1_l0b\": " << p.bw_l1_l0b << ",\n  \"bw_gm_ub\": " << p.bw_gm_ub
    << ",\n  \"bw_ub_gm\": " << p.bw_ub_gm << ",\n  \"hbm_aggregate_gibps\": " << p.hbm_aggregate_gibps
    << ",\n  \"l0_tile_m\": " << p.l0_tile_m << ",\n  \"l0_tile_n\": " << p.l0_tile_n
    << ",\n  \"vec_reg_bytes\": " << p.vec_reg_bytes << ",\n  \"vec_op_head\": " << p.vec_op_head
    << ",\n  \"vec_op_tail\": " << p.vec_op_tail << ",\n  \"vec_slope_pw\": " << p.vec_slope_pw
    << ",\n  \"vec_slope_reduce\": " << p.vec_slope_reduce << ",\n  \"l0_matmul_config\": {"
    << "\n    \"l0a_bytes\": " << p.l0_matmul_config.l0a_bytes
    << ",\n    \"l0b_bytes\": " << p.l0_matmul_config.l0b_bytes
    << ",\n    \"l0c_bytes\": " << p.l0_matmul_config.l0c_bytes
    << ",\n    \"min_m\": " << p.l0_matmul_config.min_m << ",\n    \"min_n\": " << p.l0_matmul_config.min_n
    << ",\n    \"min_k\": " << p.l0_matmul_config.min_k
    << ",\n    \"align_m\": " << p.l0_matmul_config.align_m
    << ",\n    \"align_n\": " << p.l0_matmul_config.align_n
    << ",\n    \"align_k\": " << p.l0_matmul_config.align_k
    << ",\n    \"allow_a_stationary\": " << (p.l0_matmul_config.allow_a_stationary ? "true" : "false")
    << ",\n    \"allow_b_stationary\": " << (p.l0_matmul_config.allow_b_stationary ? "true" : "false")
    << ",\n    \"allow_double_buffer_c\": " << (p.l0_matmul_config.allow_double_buffer_c ? "true" : "false")
    << ",\n    \"allow_padding\": " << (p.l0_matmul_config.allow_padding ? "true" : "false")
    << ",\n    \"allow_k_boundary\": " << (p.l0_matmul_config.allow_k_boundary ? "true" : "false")
    << ",\n    \"bw_l0a\": " << p.l0_matmul_config.bw_l0a
    << ",\n    \"bw_l0b\": " << p.l0_matmul_config.bw_l0b
    << ",\n    \"bw_drain\": " << p.l0_matmul_config.bw_drain
    << ",\n    \"bw_l0c_l1\": " << p.l0_matmul_config.bw_l0c_l1
    << ",\n    \"drain_fixed_cycles\": " << p.l0_matmul_config.drain_fixed_cycles
    << ",\n    \"drain_row_cycles\": " << p.l0_matmul_config.drain_row_cycles
    << ",\n    \"drain_penalty_cycles\": " << p.l0_matmul_config.drain_penalty_cycles
    << ",\n    \"drain_c0_bytes\": " << p.l0_matmul_config.drain_c0_bytes
    << ",\n    \"mad_head_cycles\": " << p.l0_matmul_config.mad_head_cycles
    << ",\n    \"mad_k_fractal_bytes\": " << p.l0_matmul_config.mad_k_fractal_bytes << "\n  }"
    << ",\n  \"require_uniform_cube_dag_grid\": " << (p.require_uniform_cube_dag_grid ? "true" : "false")
    << ",\n  \"use_hierarchical_cube_cost\": " << (p.use_hierarchical_cube_cost ? "true" : "false")
    << "\n}\n";
}

// Dump the solver's DECISION (fusion groups + per-group tile/latency/retain) as
// JSON for `3rdparty/pto-fusebox/scripts/visualize.py solution <dag.json> <sol.json>`.
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
  // Solver-owned per-core vector streaming algorithm for this exact candidate.
  // It is authoritative for reduction materialize-vs-stream, chunk geometry,
  // pass count, and loop kinds.
  ::VectorStreamPlan vector_stream;
  // Solver-owned cube grid, resident-band, and per-op K-loop algorithm for the
  // final/forced configuration. Candidate costs keep this out of CostResult and
  // re-derive it once here, exactly like VectorStreamPlan.
  ::CubeSchedulePlan cube_schedule;
  // Solver-owned cross-engine stage/FIFO/loop descriptor. It is reconstructed
  // only for the selected or forced candidate; the local-search CostResult
  // remains scalar and compact.
  ::MixedSchedulePlan mixed_schedule;

  SolverTile() = default;
  SolverTile(int64_t w, int64_t h, int64_t k, size_t split,
             int64_t parts_m, int64_t parts_n,
             const ::VectorStreamPlan& vector_stream,
             ::CubeSchedulePlan cube_schedule,
             ::MixedSchedulePlan mixed_schedule = {})
      : w(w), h(h), k(k), split(static_cast<int64_t>(split)),
        parts_m(parts_m), parts_n(parts_n),
        vector_stream(vector_stream),
        cube_schedule(std::move(cube_schedule)),
        mixed_schedule(std::move(mixed_schedule)) {}
};

static const char* VectorStreamKindName(::VectorStreamKind kind) {
  switch (kind) {
    case ::VectorStreamKind::Materialized:
      return "materialized";
    case ::VectorStreamKind::Pointwise:
      return "pointwise";
    case ::VectorStreamKind::ReductionFolded:
      return "reduction_folded";
    case ::VectorStreamKind::ReductionSpanning:
      return "reduction_spanning";
    case ::VectorStreamKind::SoftmaxFlash:
      return "softmax_flash";
    case ::VectorStreamKind::LayerNormWelford:
      return "layernorm_welford";
    case ::VectorStreamKind::ModelAheadMultiReduction:
      return "model_ahead_multi_reduction";
  }
  return "unknown";
}

ExprPtr MakeTuple2(ExprPtr a, ExprPtr b, const Span& sp) {
  return std::make_shared<MakeTuple>(std::vector<ExprPtr>{std::move(a), std::move(b)}, sp);
}

ExprPtr AddIndexOffset(const ExprPtr& base, const ExprPtr& delta, const Span& sp) {
  if (auto value = As<ConstInt>(base); value != nullptr && value->value_ == 0) return delta;
  if (auto value = As<ConstInt>(delta); value != nullptr && value->value_ == 0) return base;
  return MakeAdd(base, delta, sp);
}

DataType CubeAccumulatorDType(const DataType& input_dtype) {
  return input_dtype.IsFloat() ? DataType::FP32 : DataType::INT32;
}

bool MixedCubeOperandDTypeSupported(const DataType& dtype) {
  return dtype == DataType::FP32 || dtype == DataType::FP16 ||
         dtype == DataType::BF16;
}

bool CubeOutputCanCarryKLoop(const ExprPtr& lhs, const DataType& output_dtype) {
  auto lhs_type = AsTensorTypeLike(lhs->GetType());
  return lhs_type != nullptr && CubeAccumulatorDType(lhs_type->dtype_) == output_dtype;
}

utils::Stationarity FromFuseboxL0Stationarity(::L0Stationarity stationarity) {
  switch (stationarity) {
    case ::L0Stationarity::Output:
      return utils::Stationarity::kOutputStationary;
    case ::L0Stationarity::A:
      return utils::Stationarity::kAStationary;
    case ::L0Stationarity::B:
      return utils::Stationarity::kBStationary;
  }
  INTERNAL_UNREACHABLE << "AutoFuse: unknown PTO Fusebox L0 stationarity";
}

utils::L0PlanOutputTarget FromFuseboxL0OutputTarget(::L0OutputTarget target) {
  switch (target) {
    case ::L0OutputTarget::Acc:
      return utils::L0PlanOutputTarget::kAcc;
    case ::L0OutputTarget::GM:
      return utils::L0PlanOutputTarget::kGM;
    case ::L0OutputTarget::L1:
      return utils::L0PlanOutputTarget::kL1;
  }
  INTERNAL_UNREACHABLE << "AutoFuse: unknown PTO Fusebox L0 output target";
}

int64_t ExprDTypeBytes(const ExprPtr& expr) {
  auto type = AsTensorTypeLike(expr->GetType());
  INTERNAL_CHECK(type) << "AutoFuse: L0-plan matmul operand is not tensor-like";
  return std::max<int64_t>(1, static_cast<int64_t>(type->dtype_.GetBit()) / 8);
}

ExprPtr AttachL0MatmulPlan(const ExprPtr& expr, const ::L0MatmulPlan* plan, int64_t source_m,
                           int64_t source_n, int64_t source_k, const ExprPtr& lhs, const ExprPtr& rhs,
                           const DataType& output_dtype, bool accumulator_read) {
  if (plan == nullptr) return expr;
  INTERNAL_CHECK(plan->feasible) << "AutoFuse: cannot attach an infeasible L0 matmul child plan";
  auto call = As<Call>(expr);
  INTERNAL_CHECK(call && (IsOp(call, "tensor.matmul") || IsOp(call, "tensor.matmul_acc")))
      << "AutoFuse: L0 child plan must annotate tensor.matmul or tensor.matmul_acc";

  auto attrs = call->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(), [](const auto& item) {
                return item.first == utils::kL0MatmulPlanAttr ||
                       item.first == utils::kL0MatmulOutputTargetAttr;
              }),
              attrs.end());
  attrs.emplace_back(utils::kL0MatmulOutputTargetAttr,
                     static_cast<int>(FromFuseboxL0OutputTarget(plan->output_target)));

  // Analytic mode delegates the detailed L0 optimization to AutoTileMatmulL0.
  // The shared chooser is still used once while reconstructing the winning
  // GM/L1 schedule, to select a legal Acc-resident output micro-tile, but its
  // geometry is not part of the outer solver decision and is not pinned here.
  if (!ExactL0CostEnabled()) {
    return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs),
                                  call->GetType(), call->span_);
  }

  utils::L0MatmulPlanRecord record;
  record.source_m = source_m;
  record.source_n = source_n;
  record.source_k = source_k;
  record.bytes_a = ExprDTypeBytes(lhs);
  record.bytes_b = ExprDTypeBytes(rhs);
  record.bytes_c = std::max<int64_t>(1, static_cast<int64_t>(output_dtype.GetBit()) / 8);
  record.accumulator_read = accumulator_read;
  record.output_target = FromFuseboxL0OutputTarget(plan->output_target);
  record.tile_m = plan->m;
  record.tile_n = plan->n;
  record.tile_k = plan->k;
  record.stationarity = FromFuseboxL0Stationarity(plan->stationarity);
  record.output_stationary_holds_a = plan->output_stationary_holds_a;
  record.buffer_depth_a = plan->buffer_depth_a;
  record.buffer_depth_b = plan->buffer_depth_b;
  record.buffer_depth_c = plan->buffer_depth_c;
  record.k_full_chunks = plan->k_loop.full_chunks;
  record.k_tail = plan->k_loop.tail;
  record.k_pipeline_stages = plan->k_loop.pipeline_stages;
  record.estimated_traffic_bytes = plan->estimated_traffic_bytes;
  record.estimated_cost_cycles = plan->estimated_cost_cycles;
  record.padded_compute_volume = plan->padded_compute_volume;

  attrs.emplace_back(utils::kL0MatmulPlanAttr, utils::EncodeL0MatmulPlanRecord(record));
  return std::make_shared<Call>(call->op_, call->args_, call->kwargs_, std::move(attrs), call->GetType(),
                                call->span_);
}

// Build the compute for ONE [h,w] output tile: `out = a[mi:mi+h, :] @ b[:, ni:ni+w]`,
// streaming the contraction K in `k`-strips with a stage=2 software pipeline (the
// DDR<->L1 / GM->Mat double-buffer that justifies the roofline `max(compute,ddr)`).
// Mirrors AutoTileMatmulL0's K-loop builder, but at the TENSOR level (tensor.slice
// + tensor.matmul/_acc, GM->Mat) since AutoFuse runs before ConvertTensorToTileOps.
// `mi`/`ni` are element offsets along M/N (loop vars, or constant 0). Falls back to
// a single matmul over the full K when K can't be split into >=2 clean strips.
std::vector<StmtPtr> BuildTileMatmulAt(const ExprPtr& a, const ExprPtr& b, const ExprPtr& a_row_base,
                                       const ExprPtr& a_k_base, const ExprPtr& b_k_base,
                                       const ExprPtr& b_col_base, int64_t h, int64_t w, int64_t K, int64_t k,
                                       const DataType& dtype, const VarPtr& out_var, const std::string& base,
                                       const Span& sp, int pipeline_stages = -1,
                                       const ::L0MatmulPlan* l0_init = nullptr,
                                       const ::L0MatmulPlan* l0_rolled = nullptr,
                                       const ::L0MatmulPlan* l0_tail = nullptr) {
  auto& reg = OpRegistry::GetInstance();
  const std::vector<std::pair<std::string, std::any>> mm_kw = {
      {"a_trans", false}, {"b_trans", false}, {"c_matrix_nz", false}, {"out_dtype", dtype}};
  const std::vector<std::pair<std::string, std::any>> acc_kw = {{"a_trans", false}, {"b_trans", false}};

  const int64_t num_full = (k > 0) ? K / k : 0;         // full k-strips
  const int64_t tail = (k > 0) ? K - num_full * k : K;  // ragged remainder (0 if k | K)
  const bool peel = (tail > 0);
  const bool output_can_carry = CubeOutputCanCarryKLoop(a, dtype);

  // No k-pipeline when: no k tile, fewer than 2 full strips, or a ragged tail that is not
  // a 16-fractal. We do NOT build a masked fractional-K matmul (matching AutoTileMatmulL0
  // PH-AT-007); a single matmul over the full K handles those (AutoTileMatmulL0 declines it
  // if K itself is not 16-aligned). Since the solver's k is 16-aligned, tail = K mod k is
  // 16-aligned EXACTLY when K is, so tail % 16 == 0 is the "K 16-aligned" peel gate.
  if (!output_can_carry || k <= 0 || num_full < 2 || (peel && tail % 16 != 0)) {
    // No k-pipeline (one strip): a single matmul over the full K for this tile.
    auto at =
        reg.Create("tensor.slice", {a, MakeIndexTuple({h, K}, sp), MakeTuple2(a_row_base, a_k_base, sp)}, sp);
    auto av = std::make_shared<Var>(base + "_a_t", at->GetType(), sp);
    auto bt =
        reg.Create("tensor.slice", {b, MakeIndexTuple({K, w}, sp), MakeTuple2(b_k_base, b_col_base, sp)}, sp);
    auto bv = std::make_shared<Var>(base + "_b_t", bt->GetType(), sp);
    auto mm = AttachL0MatmulPlan(reg.Create("tensor.matmul", {av, bv}, mm_kw, sp), l0_init, h, w, K, av, bv,
                                 dtype, /*accumulator_read=*/false);
    return {std::make_shared<AssignStmt>(av, at, sp), std::make_shared<AssignStmt>(bv, bt, sp),
            std::make_shared<AssignStmt>(out_var, mm, sp)};
  }

  // Legacy callers (currently the chained-matmul emitter) do not yet consume a
  // CubeSchedulePlan. Preserve their pre-plan loop exactly until the recursive
  // chain emitter is plan-driven as well. A non-negative stage count means the
  // caller supplied the concrete loop chosen by CubeSchedulePlan.
  if (pipeline_stages < 0) {
    auto acc_call = reg.Create("tensor.create", {MakeIndexTuple({h, w}, sp)},
                               {{"dtype", dtype}, {"layout", TensorLayout::ND}}, sp);
    auto acc_var = std::make_shared<Var>(base + "_acc_init", acc_call->GetType(), sp);
    auto acc_assign = std::make_shared<AssignStmt>(acc_var, acc_call, sp);
    auto ko = std::make_shared<Var>(
        base + "_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
    auto c_iter = std::make_shared<IterArg>(base + "_c", acc_var->GetType(), acc_var, sp);

    auto a_k_call = reg.Create(
        "tensor.slice",
        {a, MakeIndexTuple({h, k}, sp), MakeTuple2(a_row_base, AddIndexOffset(a_k_base, ko, sp), sp)}, sp);
    auto a_k = std::make_shared<Var>(base + "_a_k", a_k_call->GetType(), sp);
    auto a_k_assign = std::make_shared<AssignStmt>(a_k, a_k_call, sp);
    auto b_k_call = reg.Create(
        "tensor.slice",
        {b, MakeIndexTuple({k, w}, sp), MakeTuple2(AddIndexOffset(b_k_base, ko, sp), b_col_base, sp)}, sp);
    auto b_k = std::make_shared<Var>(base + "_b_k", b_k_call->GetType(), sp);
    auto b_k_assign = std::make_shared<AssignStmt>(b_k, b_k_call, sp);

    auto then_call = reg.Create("tensor.matmul", {a_k, b_k}, mm_kw, sp);
    auto then_var = std::make_shared<Var>(base + "_mm", then_call->GetType(), sp);
    auto then_assign = std::make_shared<AssignStmt>(then_var, then_call, sp);
    auto then_yield =
        std::make_shared<YieldStmt>(std::vector<ExprPtr>{then_var}, sp);
    StmtPtr then_body = SeqStmts::Flatten(
        std::vector<StmtPtr>{then_assign, then_yield}, sp);

    auto else_call =
        reg.Create("tensor.matmul_acc", {ExprPtr(c_iter), a_k, b_k}, acc_kw, sp);
    auto else_var =
        std::make_shared<Var>(base + "_mm_acc", else_call->GetType(), sp);
    auto else_assign = std::make_shared<AssignStmt>(else_var, else_call, sp);
    auto else_yield =
        std::make_shared<YieldStmt>(std::vector<ExprPtr>{else_var}, sp);
    StmtPtr else_body = SeqStmts::Flatten(
        std::vector<StmtPtr>{else_assign, else_yield}, sp);

    auto phi = std::make_shared<Var>(base + "_phi", then_call->GetType(), sp);
    auto if_stmt = std::make_shared<IfStmt>(
        MakeEq(ko, MakeIndex(0, sp), sp), then_body,
        std::optional<StmtPtr>(else_body), std::vector<VarPtr>{phi}, sp);
    auto body_yield =
        std::make_shared<YieldStmt>(std::vector<ExprPtr>{phi}, sp);
    StmtPtr body = SeqStmts::Flatten(
        std::vector<StmtPtr>{a_k_assign, b_k_assign, if_stmt, body_yield}, sp);

    std::vector<std::pair<std::string, std::any>> loop_attrs = {
        {kPipelineStagesAttr, /*stages=*/2}};
    const VarPtr loop_out = peel ? std::make_shared<Var>(base + "_kloop", then_call->GetType(), sp) : out_var;
    auto for_stmt = std::make_shared<ForStmt>(
        ko, MakeIndex(0, sp), MakeIndex(num_full * k, sp), MakeIndex(k, sp),
        std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{loop_out}, sp,
        ForKind::Pipeline, std::move(loop_attrs));
    if (!peel) return {acc_assign, for_stmt};

    const int64_t k_tail = num_full * k;
    auto at = reg.Create("tensor.slice",
                         {a, MakeIndexTuple({h, tail}, sp),
                          MakeTuple2(a_row_base, AddIndexOffset(a_k_base, MakeIndex(k_tail, sp), sp), sp)},
                         sp);
    auto av = std::make_shared<Var>(base + "_a_tl", at->GetType(), sp);
    auto bt = reg.Create("tensor.slice",
                         {b, MakeIndexTuple({tail, w}, sp),
                          MakeTuple2(AddIndexOffset(b_k_base, MakeIndex(k_tail, sp), sp), b_col_base, sp)},
                         sp);
    auto bv = std::make_shared<Var>(base + "_b_tl", bt->GetType(), sp);
    auto tail_mm =
        reg.Create("tensor.matmul_acc", {ExprPtr(loop_out), av, bv}, acc_kw, sp);
    return {acc_assign, for_stmt, std::make_shared<AssignStmt>(av, at, sp),
            std::make_shared<AssignStmt>(bv, bt, sp),
            std::make_shared<AssignStmt>(out_var, tail_mm, sp)};
  }

  // Seed the carry with the first real matmul instead of tensor.create([h,w]).
  // tensor.create lowers to a UB allocation, but a solver cube tile may exceed
  // UB/L0c and is subsequently sub-tiled. The real algorithm is: serial first
  // chunk, rolled matmul_acc loop, serial tail.
  auto a0_call =
      reg.Create("tensor.slice", {a, MakeIndexTuple({h, k}, sp), MakeTuple2(a_row_base, a_k_base, sp)}, sp);
  auto a0 = std::make_shared<Var>(base + "_a_k0", a0_call->GetType(), sp);
  auto b0_call =
      reg.Create("tensor.slice", {b, MakeIndexTuple({k, w}, sp), MakeTuple2(b_k_base, b_col_base, sp)}, sp);
  auto b0 = std::make_shared<Var>(base + "_b_k0", b0_call->GetType(), sp);
  auto first_call = AttachL0MatmulPlan(reg.Create("tensor.matmul", {a0, b0}, mm_kw, sp), l0_init, h, w, k, a0,
                                       b0, dtype, /*accumulator_read=*/false);
  auto first_var = std::make_shared<Var>(base + "_first", first_call->GetType(), sp);
  std::vector<StmtPtr> result{
      std::make_shared<AssignStmt>(a0, a0_call, sp),
      std::make_shared<AssignStmt>(b0, b0_call, sp),
      std::make_shared<AssignStmt>(first_var, first_call, sp)};

  auto ko = std::make_shared<Var>(base + "_ko", std::make_shared<ScalarType>(DataType::INDEX), sp);
  auto c_iter = std::make_shared<IterArg>(base + "_c", first_var->GetType(), first_var, sp);

  // Per-iteration k-strip slices: a[mi:mi+h, ko:ko+k], b[ko:ko+k, ni:ni+w].
  auto a_k_call = reg.Create(
      "tensor.slice",
      {a, MakeIndexTuple({h, k}, sp), MakeTuple2(a_row_base, AddIndexOffset(a_k_base, ko, sp), sp)}, sp);
  auto a_k = std::make_shared<Var>(base + "_a_k", a_k_call->GetType(), sp);
  auto a_k_assign = std::make_shared<AssignStmt>(a_k, a_k_call, sp);
  auto b_k_call = reg.Create(
      "tensor.slice",
      {b, MakeIndexTuple({k, w}, sp), MakeTuple2(AddIndexOffset(b_k_base, ko, sp), b_col_base, sp)}, sp);
  auto b_k = std::make_shared<Var>(base + "_b_k", b_k_call->GetType(), sp);
  auto b_k_assign = std::make_shared<AssignStmt>(b_k, b_k_call, sp);

  auto acc_call = AttachL0MatmulPlan(reg.Create("tensor.matmul_acc", {ExprPtr(c_iter), a_k, b_k}, acc_kw, sp),
                                     l0_rolled, h, w, k, a_k, b_k, dtype,
                                     /*accumulator_read=*/true);
  auto acc_var = std::make_shared<Var>(base + "_mm_acc", acc_call->GetType(), sp);
  auto acc_assign = std::make_shared<AssignStmt>(acc_var, acc_call, sp);
  auto body_yield = std::make_shared<YieldStmt>(std::vector<ExprPtr>{acc_var}, sp);
  StmtPtr body =
      SeqStmts::Flatten(std::vector<StmtPtr>{a_k_assign, b_k_assign, acc_assign, body_yield}, sp);

  const int64_t rolled_chunks = num_full - 1;
  const bool pipeline = pipeline_stages >= 2 && rolled_chunks >= 2;
  std::vector<std::pair<std::string, std::any>> loop_attrs;
  if (pipeline) {
    loop_attrs.emplace_back(kPipelineStagesAttr, /*stages=*/2);
    // CubeSchedulePlan pipelines GM->L1 K windows. AutoTileMatmulL0 later adds
    // the child L1->L0 pipeline from the attached L0MatmulPlan; its buffers must
    // not be multiplied by this enclosing stage count.
    loop_attrs.emplace_back(kPipelineGmToL1OnlyAttr, true);
  }
  // The rolled loop runs over the remaining FULL strips [k, num_full*k). When K
  // divides (tail==0) it binds out_var directly. When K is ragged (tail>0), the
  // loop binds an intermediate and a serial matmul_acc tail folds the last
  // [h,tail]@[tail,w] partial into it, producing out_var.
  const VarPtr loop_out =
      peel ? std::make_shared<Var>(base + "_kloop", first_call->GetType(), sp) : out_var;
  auto for_stmt = std::make_shared<ForStmt>(ko, MakeIndex(k, sp), MakeIndex(num_full * k, sp), MakeIndex(k, sp),
                                            std::vector<IterArgPtr>{c_iter}, body, std::vector<VarPtr>{loop_out},
                                            sp, pipeline ? ForKind::Pipeline : ForKind::Sequential,
                                            std::move(loop_attrs));
  result.push_back(for_stmt);
  if (!peel) return result;

  // Ragged-K tail: matmul_acc the final [h,tail]@[tail,w] partial (at K-offset num_full*k)
  // into the pipeline result. tail is 16-aligned (gated above), so this is a valid fractal
  // matmul, not a masked fractional-K one.
  const int64_t k_tail = num_full * k;  // element offset of the tail strip
  auto at = reg.Create("tensor.slice",
                       {a, MakeIndexTuple({h, tail}, sp),
                        MakeTuple2(a_row_base, AddIndexOffset(a_k_base, MakeIndex(k_tail, sp), sp), sp)},
                       sp);
  auto av = std::make_shared<Var>(base + "_a_tl", at->GetType(), sp);
  auto bt = reg.Create("tensor.slice",
                       {b, MakeIndexTuple({tail, w}, sp),
                        MakeTuple2(AddIndexOffset(b_k_base, MakeIndex(k_tail, sp), sp), b_col_base, sp)},
                       sp);
  auto bv = std::make_shared<Var>(base + "_b_tl", bt->GetType(), sp);
  auto tail_mm = AttachL0MatmulPlan(reg.Create("tensor.matmul_acc", {ExprPtr(loop_out), av, bv}, acc_kw, sp),
                                    l0_tail, h, w, tail, av, bv, dtype,
                                    /*accumulator_read=*/true);
  result.push_back(std::make_shared<AssignStmt>(av, at, sp));
  result.push_back(std::make_shared<AssignStmt>(bv, bt, sp));
  result.push_back(std::make_shared<AssignStmt>(out_var, tail_mm, sp));
  return result;
}

std::vector<StmtPtr> BuildTileMatmul(const ExprPtr& a, const ExprPtr& b, const ExprPtr& mi, const ExprPtr& ni,
                                     int64_t h, int64_t w, int64_t K, int64_t k, const DataType& dtype,
                                     const VarPtr& out_var, const std::string& base, const Span& sp,
                                     int pipeline_stages = -1) {
  return BuildTileMatmulAt(a, b, mi, MakeIndex(0, sp), MakeIndex(0, sp), ni, h, w, K, k, dtype, out_var, base,
                           sp, pipeline_stages);
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

struct CubeAccumulatorGrid {
  int64_t tile_h = 0;
  int64_t tile_w = 0;
  int64_t num_m = 0;
  int64_t num_n = 0;
};

CubeAccumulatorGrid DeriveCubeAccumulatorGrid(int64_t h, int64_t w, const DataType& dtype,
                                              int64_t max_tile_h = kL0TileM, int64_t max_tile_w = kL0TileN) {
  (void)dtype;
  // Cube accumulation is FP32 for floating inputs and INT32 for integer inputs,
  // independent of the final output dtype.
  constexpr int64_t accumulator_bytes = 4;
  const int64_t capacity = std::max<int64_t>(16LL * 16 * accumulator_bytes, ReadHwParams().cube_capacity);
  max_tile_h = std::max<int64_t>(1, max_tile_h);
  max_tile_w = std::max<int64_t>(1, max_tile_w);
  int64_t tile_h = std::min(h, max_tile_h);
  int64_t tile_w = std::min(w, max_tile_w);
  // The pto-isa base M/N dimensions are algorithmic bounds, not merely a
  // fallback when Acc bytes overflow. A wide fp16 region can fit L0c by bytes
  // while still requiring several grounded L0-N subtiles.
  if (tile_h * tile_w * accumulator_bytes > capacity) {
    int64_t best_area = 0;
    bool best_aligned = false;
    for (int64_t candidate_h = 1; candidate_h <= std::min(h, max_tile_h); ++candidate_h) {
      for (int64_t candidate_w = 1; candidate_w <= std::min(w, max_tile_w); ++candidate_w) {
        if (candidate_h * candidate_w * accumulator_bytes > capacity) continue;
        const int64_t area = candidate_h * candidate_w;
        const bool aligned = candidate_h % 16 == 0 && candidate_w % 16 == 0;
        if (area > best_area || (area == best_area && aligned && !best_aligned) ||
            (area == best_area && aligned == best_aligned && candidate_h > tile_h)) {
          best_area = area;
          best_aligned = aligned;
          tile_h = candidate_h;
          tile_w = candidate_w;
        }
      }
    }
    INTERNAL_CHECK(best_area > 0)
        << "Internal error: AutoFuse cube accumulator has no L0c base tile for region [" << h << "," << w
        << "]";
  }
  CubeAccumulatorGrid grid;
  grid.tile_h = tile_h;
  grid.tile_w = tile_w;
  grid.num_m = CeilDiv(h, tile_h);
  grid.num_n = CeilDiv(w, tile_w);
  return grid;
}

// Realize the solver's tile for a `c = tensor.matmul(a, b)`: tile the output
// `[M,N]` into `[h,w]` regions (so each fits L0c) and stream the contraction in
// `k`-strips per tile. Returns the replacement statements, or nullopt if the
// matmul is not eligible (non-default orientation / non-static shapes / tile not
// dividing the output). v0 emits Sequential output-tile loops; cross-core
// Parallel distribution of those tiles is the next increment.
std::optional<std::vector<StmtPtr>> TileMatmul(const AssignStmtPtr& assign,
                                              const SolverTile& tile,
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
  int64_t k_chunk = tile.k;
  int k_pipeline_stages = -1;
  if (tile.cube_schedule.feasible && tile.cube_schedule.matmuls.size() == 1) {
    const auto& loop = tile.cube_schedule.matmuls.front().k_loop;
    k_chunk = loop.chunk;
    k_pipeline_stages = loop.pipeline_stages;
    // tensor.matmul_acc carries floating contractions in FP32 (integers in
    // INT32). A lower-precision final output needs a FIXPIPE-aware accumulator
    // descriptor that CubeSchedulePlan does not yet represent. Decline the
    // priced K loop instead of constructing an ill-typed carry.
    if ((tile.cube_schedule.split_k > 1 || loop.full_chunks >= 2) && !CubeOutputCanCarryKLoop(a, dtype)) {
      return std::nullopt;
    }
  }

  // Clamp the output tile to the output. The grid need NOT divide the output: the
  // non-split path below tiles ceil(M/h) x ceil(N/w) with clamped (overlapping,
  // idempotent) offsets — only the split-K path requires exact division.
  int64_t h = (tile.h > 0 && tile.h < M) ? tile.h : M;
  int64_t w = (tile.w > 0 && tile.w < N) ? tile.w : N;

  // The solver tile is an L1/core work unit, not necessarily one L0c
  // accumulator. Realize the model's implicit L0 subdivision explicitly when
  // a K-streamed matmul_acc would otherwise hold the whole [h,w] region in Acc.
  // Inner accumulator tiles form a disjoint base-tile grid, which is required
  // when split-K atomically adds partials. The final row/column may be ragged;
  // emit it at its exact shape rather than requiring a divisor (which can
  // degenerate to O(h*w) one-element tiles for prime dimensions). Downstream
  // matmul lowering pads those ragged M/N edge shapes to cube fractals.
  // Transitional legacy fallback only. CubeSchedulePlan deliberately owns no
  // L0 geometry; exact plan replay delegates it to the child L0MatmulPlan and
  // AutoTileMatmulL0. Keep the pre-plan constants only for unsupported groups
  // until this fallback is retired.
  const int64_t plan_l0_m = kL0TileM;
  const int64_t plan_l0_n = kL0TileN;
  const CubeAccumulatorGrid acc_grid = DeriveCubeAccumulatorGrid(h, w, dtype, plan_l0_m, plan_l0_n);
  const int64_t acc_h = acc_grid.tile_h;
  const int64_t acc_w = acc_grid.tile_w;
  const int64_t num_acc_m = acc_grid.num_m;
  const int64_t num_acc_n = acc_grid.num_n;
  const int64_t num_acc_tiles = num_acc_m * num_acc_n;

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
    // Split-K STAYS divisor-only: the S partials atomic-ADD into a shared output tile, so
    // a ceil+clamp grid (whose ragged blocks OVERLAP the previous) would DOUBLE-COUNT the
    // overlap under the atomic add. Require exact division; a non-uniform split-K grid
    // declines to an untiled InCore scope (correct values, the parallel grid dropped).
    if (M % h != 0 || N % w != 0) {
      LOG_INFO << "AutoFuse[matmul]: split-K non-uniform grid decline — atomic-add cannot "
                  "overlap-recompute; output [" << M << "," << N << "] not divisible by tile ["
               << h << "," << w << "] (split=" << tile.split << "); runs untiled InCore";
      return std::nullopt;
    }
    const int64_t S = tile.split;
    const int64_t ksz = K / S;
    const int64_t num_m = M / h;
    const int64_t num_n = N / w;
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
    // The per-tile seed [h,w] can ITSELF exceed UB (C3's larger tiles: a [256,256] fp32 seed = 256KB >
    // the 188KB UB). Zero [M,N] in UB-FITTING [seed_h, w] tiles instead: cap the row extent so one seed
    // tile fits (a constant fill needs one live band). seed_h == h when [h,w] already fits, so aligned
    // small tiles emit the same grid as before. The grid covers [M,N] disjointly with a ragged-M clamp
    // — idempotent for the non-atomic zero fill (the matmul's atomic-add partials then land on 0).
    const auto* seed_pctx = PassContext::Current();
    const auto* seed_handler = seed_pctx ? seed_pctx->GetBackendHandler()
                                         : pypto::backend::GetBackend()->GetHandler();
    const int64_t seed_dtb = std::max<int64_t>(1, static_cast<int64_t>(dtype.GetBit()) / 8);
    const int64_t seed_ub = static_cast<int64_t>(seed_handler->GetVectorBufferCapacityBytes());
    const int64_t seed_h = std::min(h, std::max<int64_t>(1, seed_ub / std::max<int64_t>(1, w * seed_dtb)));
    const int64_t num_seed_m = (M + seed_h - 1) / seed_h;  // ceil over the M axis
    const int64_t num_seed_tiles = num_seed_m * num_n;
    // Per-seed-block tile offsets from its block index (SpmdWrap prepends st = get_block_idx()).
    auto st = std::make_shared<Var>(base + "_st", index_type, sp);
    ExprPtr s_mi = MakeMul(MakeFloorDiv(st, MakeIndex(num_n, sp), sp), MakeIndex(seed_h, sp), sp);
    if (M % seed_h != 0) s_mi = MakeMin(s_mi, MakeIndex(M - seed_h, sp), sp);  // ragged last M strip clamp
    auto s_ni = MakeMul(MakeFloorMod(st, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
    auto z_call = reg.Create("tensor.full", {MakeIndexTuple({seed_h, w}, sp), zero}, {{"dtype", dtype}}, sp);
    auto z = std::make_shared<Var>(base + "_z", z_call->GetType(), sp);
    auto seed_asm = reg.Create("tensor.assemble", {c_init, z, MakeTuple2(s_mi, s_ni, sp)}, sp);
    auto c_seeded = std::make_shared<Var>(base + "_seeded", seed_asm->GetType(), sp);
    auto seed_scope = SpmdWrap(
        st,
        std::vector<StmtPtr>{std::make_shared<AssignStmt>(z, z_call, sp),
                             std::make_shared<AssignStmt>(c_seeded, seed_asm, sp)},
        MakeIndex(num_seed_tiles, sp), name + "_seed", sp);

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
    std::vector<StmtPtr> body{std::make_shared<AssignStmt>(a_ks_v, a_ks, sp),
                              std::make_shared<AssignStmt>(b_ks_v, b_ks, sp)};
    ExprPtr assembled = c_seeded;
    int64_t acc_idx = 0;
    for (int64_t am = 0; am < num_acc_m; ++am) {
      const int64_t mo = am * acc_h;
      for (int64_t an = 0; an < num_acc_n; ++an, ++acc_idx) {
        const int64_t no = an * acc_w;
        const int64_t emit_h = std::min(acc_h, h - mo);
        const int64_t emit_w = std::min(acc_w, w - no);
        const ExprPtr ami = mo == 0 ? mi : MakeAdd(mi, MakeIndex(mo, sp), sp);
        const ExprPtr ani = no == 0 ? ni : MakeAdd(ni, MakeIndex(no, sp), sp);
        auto part_type = std::make_shared<TensorType>(
            std::vector<ExprPtr>{MakeIndex(emit_h, sp), MakeIndex(emit_w, sp)}, dtype);
        auto part = std::make_shared<Var>(base + "_part" + std::to_string(acc_idx), part_type, sp);
        for (auto& stmt : BuildTileMatmul(a_ks_v, b_ks_v, ami, ani, emit_h, emit_w,
                                          ksz, k_chunk, dtype, part,
                                          base + "_l0_" + std::to_string(acc_idx), sp,
                                          k_pipeline_stages)) {
          body.push_back(std::move(stmt));
        }

        auto asm_call = reg.Create("tensor.assemble",
                                   {assembled, part, MakeTuple2(ami, ani, sp)},
                                   {{"atomic", 1}}, sp);
        const bool last = acc_idx + 1 == num_acc_tiles;
        auto assembled_var = last
                                 ? c_var
                                 : std::make_shared<Var>(base + "_partial_out" +
                                                             std::to_string(acc_idx),
                                                         asm_call->GetType(), sp);
        body.push_back(std::make_shared<AssignStmt>(assembled_var, asm_call, sp));
        assembled = assembled_var;
      }
    }

    auto scope = SpmdWrap(t, std::move(body), MakeIndex(num_tiles * S, sp), name, sp);
    return std::vector<StmtPtr>{c_init_assign, seed_scope, scope};
  }

  // Non-split spatial grid — CEIL+CLAMP (G-A). The solver's balanced parts_m x parts_n
  // partition may have ragged regions (w/h = the MAX region extent), so the grid need not
  // divide the output. Emit num_m x num_n = ceil(M/h) x ceil(N/w) blocks (>= the priced
  // parts count, so coverage holds) and CLAMP each block's offset in-bounds below; every
  // block is then a FULL [h,w] tile whose ragged blocks OVERLAP the previous. The spatial
  // assemble is NON-atomic (a plain tile.store), so the overlap recomputes the SAME value
  // -> idempotent, numerically correct (mirrors the vector emit_strip ceil+clamp). parts
  // drives the block count so emitted blocks track the priced parts_m*parts_n; ceil <=
  // parts, so max() keeps coverage while honoring the priced grid.
  const int64_t num_m = std::max<int64_t>(tile.parts_m, CeilDiv(M, h));
  const int64_t num_n = std::max<int64_t>(tile.parts_n, CeilDiv(N, w));
  if ((tile.parts_m > 0 && num_m != tile.parts_m) || (tile.parts_n > 0 && num_n != tile.parts_n))
    LOG_INFO << "AutoFuse[matmul]: group '" << name << "' emitted grid " << num_m << "x" << num_n
             << " diverges from solver parts " << tile.parts_m << "x" << tile.parts_n
             << " (ceil > parts -> coverage-safe bump; occupancy only, same max-extent critical path)";

  // The solver's tile is the whole output: no output loop — just the k-pipeline
  // (writing directly into the original output var), wrapped in one InCore kernel.
  if (num_m == 1 && num_n == 1 && num_acc_tiles == 1) {
    auto stmts = BuildTileMatmul(a, b, MakeIndex(0, sp), MakeIndex(0, sp), M, N, K,
                                 k_chunk, dtype, c_var, base, sp, k_pipeline_stages);
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
  // Clamp the ceil-grid offsets in-bounds: a ragged (or over-tiled parts>ceil) block's raw
  // offset mt*h can exceed M-h, so pin it to M-h -> the block reads a full [h,w] tile that
  // OVERLAPS the previous. The spatial assemble is non-atomic (tile.store), so the overlap
  // recomputes the same value (idempotent). A grid that divides exactly (num_m*h == M) skips
  // the clamp -> byte-identical to the pre-ceil emit.
  ExprPtr mi = MakeMul(MakeFloorDiv(t, MakeIndex(num_n, sp), sp), MakeIndex(h, sp), sp);
  ExprPtr ni = MakeMul(MakeFloorMod(t, MakeIndex(num_n, sp), sp), MakeIndex(w, sp), sp);
  if (num_m * h > M) mi = MakeMin(mi, MakeIndex(M - h, sp), sp);
  if (num_n * w > N) ni = MakeMin(ni, MakeIndex(N - w, sp), sp);
  // Per-tile body: compute the [h,w] tile (k-pipeline) and assemble it into the shared
  // output. Each SPMD core runs ONE tile (selected by get_block_idx, prepended by SpmdWrap)
  // and writes its disjoint [h,w] region, binding c_var (no IterArg/Yield -- data-parallel).
  std::vector<StmtPtr> body_stmts;
  ExprPtr assembled = c_init;
  int64_t acc_idx = 0;
  for (int64_t am = 0; am < num_acc_m; ++am) {
    const int64_t mo = am * acc_h;
    for (int64_t an = 0; an < num_acc_n; ++an, ++acc_idx) {
      const int64_t no = an * acc_w;
      const int64_t emit_h = std::min(acc_h, h - mo);
      const int64_t emit_w = std::min(acc_w, w - no);
      const ExprPtr ami = mo == 0 ? mi : MakeAdd(mi, MakeIndex(mo, sp), sp);
      const ExprPtr ani = no == 0 ? ni : MakeAdd(ni, MakeIndex(no, sp), sp);
      auto tile_type = std::make_shared<TensorType>(
          std::vector<ExprPtr>{MakeIndex(emit_h, sp), MakeIndex(emit_w, sp)}, dtype);
      auto tile_var =
          std::make_shared<Var>(base + "_tile" + std::to_string(acc_idx), tile_type, sp);
      for (auto& stmt : BuildTileMatmul(a, b, ami, ani, emit_h, emit_w, K, k_chunk,
                                        dtype, tile_var,
                                        base + "_l0_" + std::to_string(acc_idx), sp,
                                        k_pipeline_stages)) {
        body_stmts.push_back(std::move(stmt));
      }
      auto asm_call = reg.Create("tensor.assemble",
                                 {assembled, tile_var, MakeTuple2(ami, ani, sp)}, sp);
      const bool last = acc_idx + 1 == num_acc_tiles;
      auto assembled_var = last
                               ? c_var
                               : std::make_shared<Var>(base + "_assembled" +
                                                           std::to_string(acc_idx),
                                                       asm_call->GetType(), sp);
      body_stmts.push_back(std::make_shared<AssignStmt>(assembled_var, asm_call, sp));
      assembled = assembled_var;
    }
  }

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
std::optional<std::vector<StmtPtr>> TilePointwiseGroup(const std::vector<StmtPtr>& run,
                                                      const SolverTile& tile,
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

static bool MixedEmitEnabled() {
  const char* v = std::getenv("PYPTO_AUTOFUSE_MIXED");
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

static bool ExactL0CostEnabled() {
  // This is the temporary policy surface for the two optimization modes. PyPTO
  // does not yet expose a general -O2/-O3 pass-context level; once it does,
  // this flag maps naturally to that setting. Re-read per compilation just like
  // the other AutoFuse differential-testing gates.
  const char* v = std::getenv("PYPTO_AUTOFUSE_EXACT_L0_COST");
  return v != nullptr && v[0] != '\0' && std::string(v) != "0";
}

// P4 emit + cost gate. Exact online softmax is silicon-closed and therefore on
// by default. Welford remains opt-in until the extreme-shift device envelope is
// closed. An explicit legacy flag still controls both for differential/device
// runs: 0 disables all P4, while any non-empty non-zero value enables both.
// Re-read per call so tests can toggle it in-process.
static bool P4Enabled(::P4PatternKind kind) {
  const char* v = std::getenv("PYPTO_AUTOFUSE_P4");
  if (v != nullptr) return v[0] != '\0' && std::string(v) != "0";
  return kind == ::P4PatternKind::SoftmaxFlash;
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

std::optional<std::vector<StmtPtr>> EmitFusedGroupGeneric(const std::vector<StmtPtr>& run,
                                                          const SolverTile& tile,
                                                          const std::string& name,
                                                          const P4Match* p4_match,
                                                          const std::unordered_set<const Var*>&
                                                              required_live_outs) {
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
    const VectorOpDescriptor descriptor = DescribeVectorOp(c);
    const ::OpType cls = descriptor.type;
    // Tier-A: a non-{Pointwise,Reduction} op in the run -> fall back. NB a MatMul can land
    // here legitimately — a lone matmul the tiler declined (e.g. a non-uniform grid it runs
    // untiled) is pushed into `run` and reaches flush(); that is NOT a cross-engine group,
    // so it must NOT Tier-B here. Real mixed-group detection is the engine-homogeneity guard
    // at the import boundary (A2/S1), which classifies the solver's group members directly.
    if (descriptor.capability == ::VectorOpCapability::Unsupported ||
        (cls != ::OpType::Pointwise && cls != ::OpType::Reduction)) {
      return std::nullopt;
    }
    if (cls == ::OpType::Reduction) has_reduction = true;
    ops.push_back(a);
  }
  if (ops.empty()) return std::nullopt;

  // A7 (single sink): the group output is the single run-var not consumed within the run
  // (a fused group keeps its intermediates on-chip). >1 live-out = multi-output -> S5.
  std::unordered_set<const Var*> defined;
  std::unordered_map<const Var*, AssignStmtPtr> def_by_var;
  for (const auto& a : ops) {
    defined.insert(a->var_.get());
    def_by_var.emplace(a->var_.get(), a);
  }
  std::unordered_set<const Var*> used_within;
  for (const auto& a : ops) {
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      auto v = AsVarLike(arg);
      if (v != nullptr && defined.count(v.get()) != 0) used_within.insert(v.get());
    }
  }
  // Sinks = the group's live-outs (the solver's boundary outputs), in execution order.
  // A fused group MAY have >1 sink (the solver merges sinks that share input data — the
  // point of the fusion). We assemble each into its own output buffer; the shared-input
  // serialization falls out of replaying the ops in the solver's execution order.
  std::vector<AssignStmtPtr> sinks;
  for (const auto& a : ops) {
    if (used_within.count(a->var_.get()) == 0 || required_live_outs.count(a->var_.get()) != 0) {
      sinks.push_back(a);
    }
  }
  // Tier-B: a group with NO live-out is structurally impossible in SSA (every group has
  // an output; a cycle among run ops cannot exist). If it happens the run/group mapping
  // is corrupt -> surface it.
  if (sinks.empty()) {
    return GenericDeclineB("group has no live-out (corrupt run/group mapping)", ops.front()->span_);
  }
  AssignStmtPtr out_stmt = sinks.back();  // primary sink (last in exec order): shape/base for
                                          // the single-sink pipeline/split/serial paths

  // S2 (reduction-sink split): the solver may gang S cores per spatial tile, each reducing
  // a SLICE of the reduced axis, the S partials atomic-merged. Realized below for a SUM
  // col-reduction sink (the only lowered AtomicType is kAdd, and emit_strip slices rows =
  // the reduced axis of a col reduction). The solver-owned VectorStreamPlan admits only
  // that exact protocol; any other split>1 is a Tier-B contract failure rather than a
  // silent serial fallback.

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

  // Operand map over the iteration space: every operand is an intermediate, an [IM,IN] external
  // input (sliced), a BROADCAST external input (each axis is either the full extent OR 1 — the
  // FIXED_1 read-in-full role, §3/A3: `[1,IN]` M-broadcast bias, `[IM,1]` N-broadcast scale / a
  // reduced-axis stat), or a scalar. Any OTHER 2D shape is not a clean broadcast -> out of scope.
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
      if (aM == IM && aN == IN) continue;       // full external input -> sliced [sh,sw]
      // Broadcast operand: each axis is full (follows the tile) or 1 (read whole, broadcast).
      // emit_strip slices it [aM==1?1:sh, aN==1?1:sw] at [aM==1?0:smi, aN==1?0:sni]; the op replay
      // re-infers the broadcast result. Excludes [IM,IN] (handled above) and any ragged 2D shape.
      if ((aM == 1 || aM == IM) && (aN == 1 || aN == IN)) continue;  // broadcast -> sliced per-axis
      return std::nullopt;                      // other 2D shape -> out of scope
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
  const ::VectorStreamPlan& solver_stream = tile.vector_stream;
  const bool has_planned_grid = solver_stream.feasible && solver_stream.work_units > 0;
  int64_t h = pin_m ? IM
                    : (has_planned_grid ? solver_stream.tile_h
                                        : ((tile.h > 0 && tile.h < IM) ? tile.h : IM));
  int64_t w = pin_n ? IN
                    : (has_planned_grid ? solver_stream.tile_w
                                        : ((tile.w > 0 && tile.w < IN) ? tile.w : IN));
  const int64_t num_m =
      has_planned_grid ? solver_stream.m_partition.parts : (IM + h - 1) / h;
  const int64_t num_n =
      has_planned_grid ? solver_stream.n_partition.parts : (IN + w - 1) / w;

  // P1 STREAMED REDUCTION — decide whether the pinned reduced-axis tile overflows UB. When it does,
  // the reduced axis cannot be materialized in one tile; stream it (SPMD over the FREE axis, inner
  // chunk-accumulation loop over the pinned axis, persisting only the small [.,1]/[1,.] accumulator).
  // The realized streamed emit is below (after emit_strip). P1 scope: exactly one reduced axis; a
  // SINGLE reduction sink that is sum or max; the ONLY reduction (single level — a pre-reduction
  // pointwise is recomputed per chunk, but no reduction may feed a pointwise/another reduction, which
  // is P2/P3). fp32 accumulation and the ragged tail are handled in the emit.
  const auto* pctx = PassContext::Current();
  const auto* handler = pctx ? pctx->GetBackendHandler() : pypto::backend::GetBackend()->GetHandler();
  INTERNAL_CHECK(handler) << "Internal error: BackendHandler is null in AutoFuse generic emit";
  const int64_t p1_dtb = std::max<int64_t>(1, static_cast<int64_t>(dtype.GetBit()) / 8);
  const int64_t p1_ub = static_cast<int64_t>(handler->GetVectorBufferCapacityBytes());
  // Vector DMA-block granule (elements): a tile's contiguous-axis byte extent must be a multiple of
  // GetVectorDmaAlignmentBytes() (32), so the emit allocates AlignUp(extent, g)-padded tiles. Use
  // the MAX granule over the group's dtypes (= smallest dtype_bytes): a mixed FP16/FP32 chain must
  // satisfy FP16's 16-element block, which also satisfies FP32's 8. FP32 -> 8, FP16 -> 16. Computed
  // HERE (not only at the emit below) because the materialize-vs-stream trigger must count the
  // PADDED tile — the same footprint the cost model's vector_peak_ub prices. Without it a thin free
  // axis (an M-tile of 3 -> 8, ~2.7x) is under-counted, materializes an over-UB tile, and overflows
  // AllocateMemoryAddr (BUG-G1THRESH). See ascend910b_cost.cpp vector_peak_ub for the model side.
  int64_t min_dtype_bits = std::max<int64_t>(1, static_cast<int64_t>(dtype.GetBit()));
  for (const auto& a : ops) {
    if (auto tt = As<TensorType>(a->var_->GetType())) {
      min_dtype_bits =
          std::min(min_dtype_bits, std::max<int64_t>(1, static_cast<int64_t>(tt->dtype_.GetBit())));
    }
    for (const ExprPtr& arg : As<Call>(a->value_)->args_) {
      if (auto att = As<TensorType>(arg->GetType())) {
        min_dtype_bits =
            std::min(min_dtype_bits, std::max<int64_t>(1, static_cast<int64_t>(att->dtype_.GetBit())));
      }
    }
  }
  const int64_t min_dtype_bytes = std::max<int64_t>(1, (min_dtype_bits + 7) / 8);
  const int64_t g =
      std::max<int64_t>(1, handler->GetVectorDmaAlignmentBytes() / min_dtype_bytes);
  int p1_nreds = 0;
  bool p1_red_sum_or_max = false;  // the single reduction's family (sum/max = the on-core merges)
  for (const auto& a : ops)
    if (ClassifyOp(As<Call>(a->value_)) == ::OpType::Reduction) {
      p1_nreds++;
      auto rc = As<Call>(a->value_);
      p1_red_sum_or_max = IsOp(rc, "tensor.col_sum") || IsOp(rc, "tensor.row_sum") ||
                          IsOp(rc, "tensor.col_max") || IsOp(rc, "tensor.row_max");
    }
  auto p1_sink = As<Call>(out_stmt->value_);
  const bool sink_is_reduction = p1_sink != nullptr && ClassifyOp(p1_sink) == ::OpType::Reduction;
  // Common gate: exactly one reduced axis, one sum/max reduction (single level). P1 = every live-out
  // stays folded on that axis (a bare reduction or a thin pointwise finalize). P2 = a pointwise sink
  // consumes the reduction and spans the reduced axis (rmsnorm / x-row_max), so the final apply
  // chunks it. More than one reduction / level >= 2 requires an exact P4 pattern.
  // Materialize-vs-stream and the GRANULE-PADDED scratch peak are solver-owned. The structural
  // predicates below are defense-in-depth checks that this group can realize the selected plan.
  // P4 is admitted only when this solver group exactly equals a semantic match produced once by
  // AnalyzeP4Patterns. No family/count heuristic is repeated in the emitter.
  const ::P4PatternKind p4_kind = p4_match != nullptr ? p4_match->kind : ::P4PatternKind::None;
  // The solver plan is authoritative for materialize-vs-stream. The old emitter made this decision
  // from a local two-band estimate, which could disagree with the model's pebbling peak (and then
  // realize a different split/roofline). Structural checks remain as defense in depth.
  const bool plan_streams = solver_stream.feasible && solver_stream.streamed();
  const bool single_reduction_streamable =
      has_reduction && (pin_m != pin_n) && sinks.size() == 1 && p1_nreds == 1 && p1_red_sum_or_max;
  const bool stream_p1 = plan_streams && solver_stream.kind == ::VectorStreamKind::ReductionFolded &&
                         single_reduction_streamable;
  const bool stream_p2 = plan_streams && solver_stream.kind == ::VectorStreamKind::ReductionSpanning &&
                         single_reduction_streamable && !sink_is_reduction;
  const bool stream_p4 =
      plan_streams && P4Enabled(p4_kind) && p4_kind != ::P4PatternKind::None && has_reduction &&
      (pin_m != pin_n) && sinks.size() == 1 && p1_nreds >= 2 &&
      ((p4_kind == ::P4PatternKind::SoftmaxFlash &&
        solver_stream.kind == ::VectorStreamKind::SoftmaxFlash) ||
       (p4_kind == ::P4PatternKind::LayerNormWelford &&
        solver_stream.kind == ::VectorStreamKind::LayerNormWelford));
  if (has_reduction && plan_streams && !stream_p1 && !stream_p2 && !stream_p4) {
    return GenericDeclineB("solver selected an unsupported vector stream algorithm ('" +
                               std::string(VectorStreamKindName(solver_stream.kind)) + "')",
                           out_stmt->span_);
  }

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

  // Flat SPMD grid over the solver's logical regions; block -> (mi,ni).
  // `partition_offset` reconstructs the balanced ownership chosen by the model.
  // Every block still compiles the maximum region shape, so the final smaller
  // region clamps back by at most one granule and idempotently overlaps its
  // predecessor. UB/DMA padding is handled below and never changes the grid.
  auto t = std::make_shared<Var>(base + "_t", index_type, sp);
  auto partition_offset = [&](const ExprPtr& index, const ::AxisPartition& partition) {
    ExprPtr offset = MakeMul(index, MakeIndex(partition.small, sp), sp);
    const int64_t extra = partition.big - partition.small;
    if (extra > 0 && partition.num_big > 0) {
      offset = MakeAdd(offset,
                       MakeMul(MakeMin(index, MakeIndex(partition.num_big, sp)),
                               MakeIndex(extra, sp), sp),
                       sp);
    }
    return offset;
  };
  const ExprPtr m_index = MakeFloorDiv(t, MakeIndex(num_n, sp), sp);
  const ExprPtr n_index = MakeFloorMod(t, MakeIndex(num_n, sp), sp);
  ExprPtr mi = has_planned_grid
                   ? partition_offset(m_index, solver_stream.m_partition)
                   : MakeMul(m_index, MakeIndex(h, sp), sp);
  ExprPtr ni = has_planned_grid
                   ? partition_offset(n_index, solver_stream.n_partition)
                   : MakeMul(n_index, MakeIndex(w, sp), sp);
  if (num_m * h > IM) mi = MakeMin(mi, MakeIndex(IM - h, sp), sp);
  if (num_n * w > IN) ni = MakeMin(ni, MakeIndex(IN - w, sp), sp);

  // Vector DMA-block granule `g` (elements) is computed above (before the stream trigger) so the
  // materialize-vs-stream decision counts the padded tile. Reduced-axis padding is now ALLOWED. The original §4.4 concern — a reduction over a ragged
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
  // slice_input: slice ONE 2D external input `arg` to a [sh, sw] VALID region at (smi, sni),
  // granule-padding the ALLOCATED extent (the row axis padded only for a reduction/col-major tile —
  // see the layout note in emit_strip). A broadcast axis (extent 1 in `arg`) stays [1] at offset 0
  // (read whole, broadcast in the op). Pushes the slice assign to `out` (named base+"_in"+slot so
  // distinct inputs never collapse) and returns its Var. Factored out of emit_strip so the P4 pass-0
  // online-stats body can DMA a chunk's x-slice EXACTLY ONCE — the io_in*=2 (A7 stream_passes=2)
  // pricing depends on each chunk being read once per pass. For a full [IM,IN] input with an aligned
  // region this is byte-identical to emit_strip's prior inline slice.
  auto slice_input = [&](const ExprPtr& arg, int64_t sh, int64_t sw, const ExprPtr& smi,
                         const ExprPtr& sni, std::vector<StmtPtr>& out, int slot) -> VarPtr {
    const auto [aM, aN] = Static2DShape(arg->GetType());
    const int64_t sh_al = has_reduction ? AlignUp(sh, g) : sh;
    const int64_t sw_al = AlignUp(sw, g);
    const bool bcast_m = (aM == 1), bcast_n = (aN == 1);
    const int64_t rext = bcast_m ? 1 : sh, rext_al = bcast_m ? 1 : sh_al;
    const int64_t cext = bcast_n ? 1 : sw, cext_al = bcast_n ? 1 : sw_al;
    const ExprPtr roff = bcast_m ? MakeIndex(0, sp) : smi;
    const ExprPtr coff = bcast_n ? MakeIndex(0, sp) : sni;
    const bool sl_ragged = (rext_al != rext) || (cext_al != cext);
    ExprPtr sl = sl_ragged
                     ? reg.Create("tensor.slice",
                                  {arg, MakeIndexTuple({rext_al, cext_al}, sp), MakeTuple2(roff, coff, sp),
                                   MakeIndexTuple({rext, cext}, sp)},
                                  sp)
                     : reg.Create("tensor.slice", {arg, MakeIndexTuple({rext, cext}, sp),
                                                   MakeTuple2(roff, coff, sp)}, sp);
    auto sv = std::make_shared<Var>(base + "_in" + std::to_string(slot), sl->GetType(), sp);
    out.push_back(std::make_shared<AssignStmt>(sv, sl, sp));
    return sv;
  };

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
  // Axis-symmetric slice-and-replay (streamed-reduction §4): slice each full external input to a
  // [sh, sw] region at (smi, sni) — BOTH extents are parameters, so a caller can chunk the row axis
  // (S2 split / col-reduction stream) OR the col axis (row-reduction stream), not just rows. Spatial
  // and pipeline callers pass sw = w (the full tile width) → byte-identical to the pre-refactor form.
  // `stop_at` (default = out_stmt): replay ops up to AND INCLUDING this op, return its tile, then
  //   stop — so a streamed pass can stop at the REDUCTION (P2 pass 0) instead of the group sink.
  //   nullptr keeps the pre-P2 behavior (replay every op; multi-sink reads all sinks from onchip).
  // `subs` (op-var -> finalized-accumulator tile): when an op's OUTPUT is in `subs`, do NOT replay
  //   it — bind the substitute tile (P2 pass 1 uses the finalized reduction result instead of
  //   recomputing a partial). This is the value-level "substitute reductions at level < k" rule.
  auto emit_strip = [&](int64_t sh, int64_t sw, const ExprPtr& smi, const ExprPtr& sni,
                        std::vector<StmtPtr>& out, std::unordered_map<const Var*, VarPtr>& onchip,
                        const Stmt* stop_at = nullptr,
                        const std::unordered_map<const Var*, VarPtr>* subs = nullptr) -> VarPtr {
    const Stmt* sink_op = stop_at != nullptr ? stop_at : out_stmt.get();
    // Emit exactly the dependency cone of this phase.  A substituted statistic
    // is a leaf: bind its supplied tile, but do not replay its producer cone.
    // Multi-sink replay deliberately keeps all ops because every sink is read
    // from `onchip` by its caller.
    std::unordered_set<const Stmt*> needed;
    const bool prune_to_sink = sinks.size() == 1 || stop_at != nullptr || subs != nullptr;
    if (prune_to_sink) {
      std::vector<AssignStmtPtr> stack;
      for (const auto& candidate : ops)
        if (candidate.get() == sink_op) {
          stack.push_back(candidate);
          break;
        }
      while (!stack.empty()) {
        AssignStmtPtr current = stack.back();
        stack.pop_back();
        if (!needed.insert(current.get()).second) continue;
        if (subs != nullptr && subs->count(current->var_.get()) != 0) continue;
        for (const ExprPtr& arg : As<Call>(current->value_)->args_) {
          auto var = AsVarLike(arg);
          if (var == nullptr) continue;
          auto producer = def_by_var.find(var.get());
          if (producer != def_by_var.end()) stack.push_back(producer->second);
        }
      }
    }
    // The 32B DMA granule is on the CONTIGUOUS axis only; the other (free) axis has
    // granule 1 (see ascend910b_cost.cpp: "free row axis tiles at 1 element, the
    // contiguous width axis at the 32-byte DMA block"). So pad the row axis ONLY when
    // rows are contiguous — i.e. the tile is col-major, which a reduction group is (ptoas
    // treats softmax/norm tiles as col-major none_box: "column byte size (rows*dtype)" must
    // be 32-aligned, so the ROWS are the contiguous axis and DO need padding — verified by
    // the ptoas assembly gate). A pure pointwise tile is row-major (cols contiguous) → rows
    // are the FREE axis and need no padding; padding them is the [64,4096] over-fetch that
    // overflows UB. `has_reduction` is an interim proxy for the tile layout, which is not
    // decided until InferTileMemorySpace; the layout-exact version belongs in a post-layout
    // padding pass (which also fixes the legacy tilers — KNOWN_ISSUES).
    onchip.clear();                                      // fresh per replay
    std::unordered_map<const Var*, VarPtr> input_cache;  // external input -> its [sh,w] slice
    VarPtr tv;
    for (const auto& a : ops) {
      if (prune_to_sink && needed.count(a.get()) == 0) continue;
      if (subs != nullptr) {  // finalized accumulator from an earlier pass -> substitute, don't replay
        auto sit = subs->find(a->var_.get());
        if (sit != subs->end()) {
          onchip[a->var_.get()] = sit->second;
          if (a.get() == sink_op) { tv = sit->second; break; }
          continue;
        }
      }
      auto c = As<Call>(a->value_);
      std::vector<ExprPtr> targs;
      for (const ExprPtr& arg : c->args_) {
        auto v = AsVarLike(arg);
        if (v != nullptr) {
          auto it = onchip.find(v.get());
          if (it != onchip.end()) { targs.push_back(it->second); continue; }  // intermediate on-chip
        }
        const auto [aM, aN] = Static2DShape(arg->GetType());
        if (aM < 0) {
          targs.push_back(arg);  // scalar / non-2D -> as-is
          continue;
        }
        // 2D external input (full [IM,IN] or a broadcast [1,IN]/[IM,1], validated at the group top).
        // Slice per-axis: a FULL axis follows the tile (offset + granule-padded alloc + ragged valid);
        // a size-1 (broadcast, FIXED_1) axis stays [1] at offset 0 (read whole, broadcast in the op).
        // Cached per input var. For a full [IM,IN] input this is byte-identical to the prior form.
        if (v != nullptr) {
          auto sit = input_cache.find(v.get());
          if (sit != input_cache.end()) { targs.push_back(sit->second); continue; }
        }
        // Unique name per distinct external input; a name-based consumer must not collapse >1 input.
        VarPtr sv = slice_input(arg, sh, sw, smi, sni, out, static_cast<int>(input_cache.size()));
        if (v != nullptr) input_cache[v.get()] = sv;
        targs.push_back(sv);
      }
      auto pw = reg.Create(c->op_->name_, targs, c->kwargs_, sp);
      // Unique name per intermediate (a multi-consumer intermediate must keep a distinct name).
      auto res = std::make_shared<Var>(
          a == out_stmt ? (base + "_tile") : (base + "_t" + std::to_string(onchip.size())),
          pw->GetType(), sp);
      out.push_back(std::make_shared<AssignStmt>(res, pw, sp));
      onchip[a->var_.get()] = res;
      if (a.get() == sink_op) {
        tv = res;
        if (stop_at != nullptr) break;  // stop after the designated sink (P1/P2); else replay all
      }
    }
    return tv;
  };

  // P1 STREAMED REDUCTION (realize the decision made above). The reduced axis is too large to
  // materialize; stream it. SPMD over the FREE axis; each core runs an inner chunk-accumulation
  // loop over the pinned axis, persisting only the small reduced [.,1]/[1,.] accumulator (the big
  // [.,chunk] slices are transient per iteration). Accumulation is ON-CORE (ordinary tile add/max,
  // NOT the cross-core atomic), so it is exact for sum AND max. Chunks are DISJOINT (no clamp-
  // overlap: a reduction overlap would double-count), ragged tail via `valid`. The accumulator loop
  // uses an iter_arg for the persistent accumulator — lowering-proven (the §11.3 spike: MemoryReuse
  // aliases the carry in place) — and pipelines its full-chunk loop when there are two rolled
  // iterations to overlap. Single pass (level-0 reduction); P2/P4 add an apply re-stream.
  if (stream_p1 || stream_p2 || stream_p4) {
    const int64_t red_ext = pin_m ? IM : IN;    // pinned/reduced axis extent
    // The free tile is the maximum LOGICAL region extent. Its UB allocation is
    // independently DMA-padded; alignment must not coarsen the logical launch
    // grid. This is the model/emit distinction that preserves parts_m/parts_n.
    const int64_t expected_free_tile = pin_m ? w : h;
    const int64_t expected_free_alloc = AlignUp(expected_free_tile, g);
    const int64_t free_tile = solver_stream.free_tile;
    // The single reduction op (nreds==1): P1 -> every live-out stays folded; P2 -> a non-sink
    // reduction consumed by a spanning pointwise sink. Its family fixes the merge op
    // (sum->add, max->maximum).
    auto red_stmt = out_stmt;  // AssignStmtPtr (same element type as ops)
    for (const auto& a : ops)
      if (ClassifyOp(As<Call>(a->value_)) == ::OpType::Reduction) red_stmt = a;
    auto red_call = As<Call>(red_stmt->value_);
    const bool is_max = IsOp(red_call, "tensor.col_max") || IsOp(red_call, "tensor.row_max");
    const std::string merge_op = is_max ? "tensor.maximum" : "tensor.add";
    // Chunk geometry comes from the winning solver plan. Its conservative stream-band peak includes
    // the emitted accumulator/P2/P4 scratch that is absent from the original tensor DAG, and it has
    // already proved the selected chunk fits UB. Do not independently resize it here.
    const int64_t rc = solver_stream.chunk;
    INTERNAL_CHECK_SPAN(rc > 0 && rc <= red_ext && solver_stream.axis == (pin_m ? 2 : 1), sp)
        << "Internal error: invalid solver vector stream plan for group '" << name << "': axis="
        << solver_stream.axis << " chunk=" << rc << " reduced extent=" << red_ext;
    const int64_t num_full = red_ext / rc;                 // full disjoint chunks
    const int64_t rem = red_ext - num_full * rc;           // ragged tail extent (0 if divides)
    const int64_t num_free = pin_m ? num_n : num_m;

    // Verify that the local IR construction realizes every field of the authoritative plan.
    // Geometry, algorithm kind, and the trip-count pipeline guard must all agree.
    const ::VectorStreamKind emitted_kind =
        stream_p4 ? (p4_kind == ::P4PatternKind::SoftmaxFlash
                         ? ::VectorStreamKind::SoftmaxFlash
                         : ::VectorStreamKind::LayerNormWelford)
                  : (stream_p2 ? ::VectorStreamKind::ReductionSpanning
                               : ::VectorStreamKind::ReductionFolded);
    const int emitted_axis = pin_m ? 2 : 1;
    const int64_t stats_trips = std::max<int64_t>(0, num_full - 1);
    const int expected_passes = (stream_p2 || stream_p4) ? 2 : 1;
    const bool expected_finalize = stream_p1 || p4_kind == ::P4PatternKind::LayerNormWelford;
    const ::VectorP4WorkPlan expected_p4_work = ::make_vector_p4_work_plan(p4_kind);
    const bool p4_work_matches =
        stream_p4 ? solver_stream.p4_work == expected_p4_work : !solver_stream.p4_work.generated;
    const bool stream_plan_matches =
        solver_stream.feasible && solver_stream.streamed() && solver_stream.kind == emitted_kind &&
        solver_stream.axis == emitted_axis && solver_stream.extent == red_ext &&
        solver_stream.free_tile == expected_free_tile &&
        solver_stream.free_tile_alloc == expected_free_alloc &&
        solver_stream.work_units == num_free && solver_stream.chunk == rc &&
        solver_stream.full_chunks == num_full && solver_stream.tail == rem &&
        solver_stream.stats_init.present && solver_stream.stats_init.chunk_index == 0 &&
        solver_stream.stats_init.extent == rc && solver_stream.stream_passes == expected_passes &&
        solver_stream.stats.first_chunk == 1 && solver_stream.stats.trip_count == stats_trips &&
        solver_stream.stats_tail.present == (rem > 0) &&
        (!solver_stream.stats_tail.present ||
         (solver_stream.stats_tail.chunk_index == num_full && solver_stream.stats_tail.extent == rem)) &&
        solver_stream.finalize.present == expected_finalize && p4_work_matches &&
        (expected_passes == 1 ||
         (solver_stream.apply.first_chunk == 0 && solver_stream.apply.trip_count == num_full &&
          solver_stream.apply_tail.present == (rem > 0) &&
          (!solver_stream.apply_tail.present ||
           (solver_stream.apply_tail.chunk_index == num_full && solver_stream.apply_tail.extent == rem))));
    INTERNAL_CHECK_SPAN(stream_plan_matches, sp)
        << "Internal error: emitted vector loop does not match solver plan for group '" << name << "'";
    LOG_INFO << "AutoFuse[generic]: vector-plan MATCH group='" << name
             << "' solver={kind=" << VectorStreamKindName(solver_stream.kind)
             << ",axis=" << solver_stream.axis << ",extent=" << solver_stream.extent
             << ",regions=" << solver_stream.work_units << ",free=" << solver_stream.free_tile
             << "/alloc=" << solver_stream.free_tile_alloc << ",chunk=" << solver_stream.chunk << "x"
             << solver_stream.full_chunks << (solver_stream.tail ? "+tail" : "")
             << ",stats=" << solver_stream.stats.trip_count << "/s" << solver_stream.stats.pipeline_stages
             << ",apply=" << solver_stream.apply.trip_count << "/s" << solver_stream.apply.pipeline_stages
             << "} emit={kind=" << VectorStreamKindName(emitted_kind) << ",axis=" << emitted_axis
             << ",extent=" << red_ext << ",regions=" << num_free << ",free=" << free_tile
             << "/alloc=" << expected_free_alloc << ",chunk=" << rc << "x" << num_full
             << (rem ? "+tail" : "") << ",stats=" << stats_trips << "/s"
             << solver_stream.stats.pipeline_stages << ",apply=" << (expected_passes == 2 ? num_full : 0)
             << "/s" << (expected_passes == 2 ? solver_stream.apply.pipeline_stages : 1) << "}";

    // Reuse the solver partition's balanced coordinate. The final smaller
    // logical region executes the maximum-shape body at the clamped offset;
    // this only overlaps idempotent free-axis work and remains in bounds.
    ExprPtr foff = pin_m ? ni : mi;
    // Slice the [chunk_ext along the reduced axis, free_tile] region at `red_off` and replay the
    // cone up to `stop` (nullptr = the group sink), substituting finalized accumulators from `subs`.
    auto strip_at = [&](int64_t chunk_ext, const ExprPtr& red_off, std::vector<StmtPtr>& out,
                        std::unordered_map<const Var*, VarPtr>& oc, const Stmt* stop,
                        const std::unordered_map<const Var*, VarPtr>* subs) -> VarPtr {
      // `free_tile` is the logical valid extent. `slice_input` pads the
      // allocated tile to `free_tile_alloc`, so the full region is written
      // without changing its ownership or SPMD count.
      return pin_m ? emit_strip(chunk_ext, free_tile, red_off, foff, out, oc, stop, subs)   // chunk M rows, free_tile N
                   : emit_strip(free_tile, chunk_ext, foff, red_off, out, oc, stop, subs);  // free_tile M, chunk N
    };

    // One construction for every loop-carried P2/P4 phase.  Algorithm-specific
    // code builds the carried values and YieldStmt; this helper alone translates
    // the solver's loop descriptor into Sequential vs stage-2 Pipeline IR.
    auto append_planned_loop = [&](std::vector<StmtPtr>& outer, const VarPtr& index,
                                   const ::VectorLoopPlan& loop, std::vector<IterArgPtr> iter_args,
                                   std::vector<StmtPtr> loop_body, std::vector<VarPtr> results) {
      INTERNAL_CHECK_SPAN(loop.trip_count > 0, sp)
          << "Internal error: attempted to emit an empty vector phase loop";
      const bool pipelined = loop.pipeline_stages == 2;
      std::vector<std::pair<std::string, std::any>> attrs;
      if (pipelined) {
        attrs.emplace_back(kPipelineStagesAttr, /*stages=*/2);
      }
      outer.push_back(std::make_shared<ForStmt>(
          index, MakeIndex(loop.first_chunk, sp), MakeIndex(loop.first_chunk + loop.trip_count, sp),
          MakeIndex(1, sp), std::move(iter_args), SeqStmts::Flatten(std::move(loop_body), sp),
          std::move(results), sp, pipelined ? ForKind::Pipeline : ForKind::Sequential, std::move(attrs)));
    };

    // P2, softmax, and Welford have the same barrier-separated apply algorithm:
    // replay one reduced-axis chunk with finalized statistics substituted, then
    // assemble it into the output carry.  Only the substitution map differs.
    auto emit_spanning_apply = [&](std::vector<StmtPtr>& body,
                                   const std::unordered_map<const Var*, VarPtr>& subs) {
      auto asm_at = [&](const ExprPtr& offset) -> ExprPtr {
        return pin_m ? MakeTuple2(offset, foff, sp) : MakeTuple2(foff, offset, sp);
      };
      VarPtr out_cur = c_init;
      if (solver_stream.apply.trip_count > 0) {
        auto s = std::make_shared<Var>(base + "_ps", index_type, sp);
        auto out_it = std::make_shared<IterArg>(base + "_oit", c_init->GetType(), ExprPtr(c_init), sp);
        ExprPtr offset = MakeMul(s, MakeIndex(rc, sp), sp);
        std::vector<StmtPtr> loop_body;
        std::unordered_map<const Var*, VarPtr> onchip_apply;
        VarPtr chunk_tile = strip_at(rc, offset, loop_body, onchip_apply, nullptr, &subs);
        auto assemble =
            reg.Create("tensor.assemble", {ExprPtr(out_it), ExprPtr(chunk_tile), asm_at(offset)}, sp);
        auto out_next = std::make_shared<Var>(base + "_on", assemble->GetType(), sp);
        loop_body.push_back(std::make_shared<AssignStmt>(out_next, assemble, sp));
        loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{ExprPtr(out_next)}, sp));
        auto loop_result = std::make_shared<Var>(
            solver_stream.apply_tail.present ? (base + "_ofin") : c_var->name_hint_, c_init->GetType(), sp);
        append_planned_loop(body, s, solver_stream.apply, std::vector<IterArgPtr>{out_it},
                            std::move(loop_body),
                            std::vector<VarPtr>{solver_stream.apply_tail.present ? loop_result : c_var});
        out_cur = solver_stream.apply_tail.present ? loop_result : c_var;
      }
      if (solver_stream.apply_tail.present) {
        std::unordered_map<const Var*, VarPtr> onchip_tail;
        ExprPtr offset = MakeIndex(solver_stream.apply_tail.chunk_index * rc, sp);
        VarPtr chunk_tile =
            strip_at(solver_stream.apply_tail.extent, offset, body, onchip_tail, nullptr, &subs);
        auto assemble =
            reg.Create("tensor.assemble", {ExprPtr(out_cur), ExprPtr(chunk_tile), asm_at(offset)}, sp);
        body.push_back(std::make_shared<AssignStmt>(c_var, assemble, sp));
      }
    };

    // ===================== P4 — FUSED ONLINE LAYERNORM (Welford) =====================
    // The shared descriptor has already proven the exact dual-sum layernorm algebra. The emitted stats
    // pass uses numerically stable Welford/Chan and substitutes the finalized mean and variance into
    // that exact cone. No generic independent-sum graph is allowed to reach this path.
    if (stream_p4 && p4_kind == ::P4PatternKind::LayerNormWelford) {
      INTERNAL_CHECK_SPAN(p4_match != nullptr && p4_match->layernorm_sums.size() == 2 &&
                              p4_match->user_mean != nullptr && p4_match->user_var != nullptr &&
                              p4_match->x_input != nullptr,
                          sp)
          << "Internal error: exact layernorm P4 group has an incomplete semantic descriptor";
      const VarPtr user_mean = p4_match->user_mean;
      const VarPtr user_var = p4_match->user_var;
      const ExprPtr x_input = p4_match->x_input;

      std::vector<StmtPtr> body;
      std::unordered_map<const Var*, VarPtr> subs;

      {
        // ---- WELFORD streaming (numerically STABLE). Running (mean, M2, count) as [free_tile,1] tile
        // iter_args. Count is carried as a small tile (NOT folded to a compile-time constant): the merge
        // weights cnt_a/n_new depend on the RUNTIME chunk index k in the rolled loop, so carrying count
        // keeps every merge op tile-valued (robust) instead of needing runtime-scalar operands.
        auto CFloat = [&](double v) { return std::make_shared<ConstFloat>(v, dtype, sp); };
        // welford_chunk: fold one [free_tile, chunk_ext] slice at red_off into (mean_in, M2_in, cnt_in).
        // nullptrs => the init chunk (mean=mean_a, M2=M2_a, cnt=chunk_ext). Returns (mean, M2, cnt).
        auto welford_chunk = [&](int64_t chunk_ext, const ExprPtr& red_off, const VarPtr& mean_in,
                                 const VarPtr& M2_in, const VarPtr& cnt_in, std::vector<StmtPtr>& out,
                                 int tag) -> std::tuple<VarPtr, VarPtr, VarPtr> {
          const std::string tg = std::to_string(tag);
          VarPtr xs = pin_m ? slice_input(x_input, chunk_ext, free_tile, red_off, foff, out, tag)
                            : slice_input(x_input, free_tile, chunk_ext, foff, red_off, out, tag);
          // chunk mean: mean_a = row_sum(x) / chunk_ext
          auto s_c = reg.Create("tensor.row_sum", {ExprPtr(xs)}, sp);
          auto s = std::make_shared<Var>(base + "_wsum" + tg, s_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(s, s_c, sp));
          auto ma_c = reg.Create("tensor.muls", {ExprPtr(s), CFloat(1.0 / static_cast<double>(chunk_ext))}, sp);
          auto mean_a = std::make_shared<Var>(base + "_wmeana" + tg, ma_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(mean_a, ma_c, sp));
          // chunk M2: M2_a = row_sum((x - mean_a)^2)  [stable — deviations are O(std), no cancellation]
          auto dev_c = reg.Create("tensor.row_expand_sub", {ExprPtr(xs), ExprPtr(mean_a)}, sp);
          auto dev = std::make_shared<Var>(base + "_wdev" + tg, dev_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(dev, dev_c, sp));
          auto dsq_c = reg.Create("tensor.mul", {ExprPtr(dev), ExprPtr(dev)}, sp);
          auto dsq = std::make_shared<Var>(base + "_wdsq" + tg, dsq_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(dsq, dsq_c, sp));
          auto M2a_c = reg.Create("tensor.row_sum", {ExprPtr(dsq)}, sp);
          auto M2_a = std::make_shared<Var>(base + "_wM2a" + tg, M2a_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(M2_a, M2a_c, sp));
          if (mean_in == nullptr) {  // init chunk seeds (mean, M2, count=chunk_ext)
            // Count column of value chunk_ext, derived from the reduction output `s` (NOT tensor.full):
            // s is a col_major reduction column; muls*0 + adds keeps the count column col_major, matching
            // the other stats. A fresh tile.full is row_major, which trips ResolveBackendOpLayouts'
            // col-vector layout repair (padded [free_tile,1] does not round-trip) — so avoid it.
            auto z_c = reg.Create("tensor.muls", {ExprPtr(s), CFloat(0.0)}, sp);
            auto z = std::make_shared<Var>(base + "_wz" + tg, z_c->GetType(), sp);
            out.push_back(std::make_shared<AssignStmt>(z, z_c, sp));
            auto cnt_c = reg.Create("tensor.adds", {ExprPtr(z), CFloat(static_cast<double>(chunk_ext))}, sp);
            auto cnt = std::make_shared<Var>(base + "_wcnt" + tg, cnt_c->GetType(), sp);
            out.push_back(std::make_shared<AssignStmt>(cnt, cnt_c, sp));
            return {mean_a, M2_a, cnt};
          }
          // Chan's parallel merge into the running (mean_in, M2_in, cnt_in):
          //   delta = mean_a - mean_in;  n_new = cnt_in + chunk_ext
          //   mean  = mean_in + delta*chunk_ext / n_new
          //   M2    = M2_in + M2_a + delta^2 * chunk_ext * cnt_in / n_new
          auto delta_c = reg.Create("tensor.sub", {ExprPtr(mean_a), ExprPtr(mean_in)}, sp);
          auto delta = std::make_shared<Var>(base + "_wdelta" + tg, delta_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(delta, delta_c, sp));
          auto nnew_c = reg.Create("tensor.adds", {ExprPtr(cnt_in), CFloat(static_cast<double>(chunk_ext))}, sp);
          auto n_new = std::make_shared<Var>(base + "_wnnew" + tg, nnew_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(n_new, nnew_c, sp));
          auto dm_c = reg.Create("tensor.muls", {ExprPtr(delta), CFloat(static_cast<double>(chunk_ext))}, sp);
          auto dm = std::make_shared<Var>(base + "_wdm" + tg, dm_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(dm, dm_c, sp));
          auto dmo_c = reg.Create("tensor.div", {ExprPtr(dm), ExprPtr(n_new)}, sp);
          auto dmo = std::make_shared<Var>(base + "_wdmo" + tg, dmo_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(dmo, dmo_c, sp));
          auto mean_c = reg.Create("tensor.add", {ExprPtr(mean_in), ExprPtr(dmo)}, sp);
          auto mean_new = std::make_shared<Var>(base + "_wmeann" + tg, mean_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(mean_new, mean_c, sp));
          auto d2_c = reg.Create("tensor.mul", {ExprPtr(delta), ExprPtr(delta)}, sp);
          auto d2 = std::make_shared<Var>(base + "_wd2" + tg, d2_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(d2, d2_c, sp));
          auto d2c_c = reg.Create("tensor.muls", {ExprPtr(d2), CFloat(static_cast<double>(chunk_ext))}, sp);
          auto d2c = std::make_shared<Var>(base + "_wd2c" + tg, d2c_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(d2c, d2c_c, sp));
          auto num_c = reg.Create("tensor.mul", {ExprPtr(d2c), ExprPtr(cnt_in)}, sp);
          auto num = std::make_shared<Var>(base + "_wnum" + tg, num_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(num, num_c, sp));
          auto term_c = reg.Create("tensor.div", {ExprPtr(num), ExprPtr(n_new)}, sp);
          auto term = std::make_shared<Var>(base + "_wterm" + tg, term_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(term, term_c, sp));
          auto m2s_c = reg.Create("tensor.add", {ExprPtr(M2_in), ExprPtr(M2_a)}, sp);
          auto m2s = std::make_shared<Var>(base + "_wM2s" + tg, m2s_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(m2s, m2s_c, sp));
          auto M2n_c = reg.Create("tensor.add", {ExprPtr(m2s), ExprPtr(term)}, sp);
          auto M2_new = std::make_shared<Var>(base + "_wM2n" + tg, M2n_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(M2_new, M2n_c, sp));
          return {mean_new, M2_new, n_new};
        };

        // PASS 0 — chunk 0 inits (mean, M2, count); chunks 1..num_full merge while threading
        // {mean_it, M2_it, cnt_it}; the ragged tail merges after. Chunks DISJOINT.
        auto [mean_cur, M2_cur, cnt_cur] = welford_chunk(
            solver_stream.stats_init.extent, MakeIndex(solver_stream.stats_init.chunk_index * rc, sp),
            nullptr, nullptr, nullptr, body, 0);
        if (solver_stream.stats.trip_count > 0) {
          auto k = std::make_shared<Var>(base + "_k", index_type, sp);
          auto mean_it = std::make_shared<IterArg>(base + "_wmean_it", mean_cur->GetType(), ExprPtr(mean_cur), sp);
          auto M2_it = std::make_shared<IterArg>(base + "_wM2_it", M2_cur->GetType(), ExprPtr(M2_cur), sp);
          auto cnt_it = std::make_shared<IterArg>(base + "_wcnt_it", cnt_cur->GetType(), ExprPtr(cnt_cur), sp);
          std::vector<StmtPtr> lbody;
          auto [mn, m2n, cn] =
              welford_chunk(rc, MakeMul(k, MakeIndex(rc, sp), sp), mean_it, M2_it, cnt_it, lbody, 1);
          lbody.push_back(std::make_shared<YieldStmt>(
              std::vector<ExprPtr>{ExprPtr(mn), ExprPtr(m2n), ExprPtr(cn)}, sp));
          auto mean_out = std::make_shared<Var>(base + "_wmean", mean_cur->GetType(), sp);
          auto M2_out = std::make_shared<Var>(base + "_wM2", M2_cur->GetType(), sp);
          auto cnt_out = std::make_shared<Var>(base + "_wcnt", cnt_cur->GetType(), sp);
          // A5: the Welford tuple is true loop-carried state and remains persistent. Stage=2
          // double-buffers only the next disjoint input chunk, overlapping its load with the current
          // chunk's Welford reduction/merge. Chunk 0 is emitted before this loop, so the rolled trip
          // count is num_full-1 and needs to be at least two to have anything to overlap.
          append_planned_loop(body, k, solver_stream.stats, std::vector<IterArgPtr>{mean_it, M2_it, cnt_it},
                              std::move(lbody), std::vector<VarPtr>{mean_out, M2_out, cnt_out});
          mean_cur = mean_out;
          M2_cur = M2_out;
          cnt_cur = cnt_out;
        }
        if (solver_stream.stats_tail.present) {
          auto [mn, m2n, cn] = welford_chunk(solver_stream.stats_tail.extent,
                                             MakeIndex(solver_stream.stats_tail.chunk_index * rc, sp),
                                             mean_cur, M2_cur, cnt_cur, body, 2);
          mean_cur = mn;
          M2_cur = m2n;
          cnt_cur = cn;
        }
        // Finalize: mean_final = running mean; var_final = M2 / N (population variance, matching torch
        // var(unbiased=False)). Substitute BOTH stable stats into the apply cone (mean/var level).
        auto var_c = reg.Create("tensor.muls", {ExprPtr(M2_cur), CFloat(1.0 / static_cast<double>(red_ext))}, sp);
        auto var_final = std::make_shared<Var>(base + "_wvar", var_c->GetType(), sp);
        body.push_back(std::make_shared<AssignStmt>(var_final, var_c, sp));
        subs.emplace(user_mean.get(), mean_cur);
        subs.emplace(user_var.get(), var_final);
      }

      emit_spanning_apply(body, subs);

      auto scope = SpmdWrap(t, std::move(body), MakeIndex(num_free, sp), name, sp);
      LOG_INFO << "AutoFuse[generic]: STREAMED fused online layernorm (P4) '" << name << "' ("
               << ops.size() << " ops, reduce " << (pin_m ? "M" : "N") << " ext=" << red_ext
               << " chunk=" << rc << "x" << num_full << (rem ? "+tail" : "") << ", free grid "
               << num_free << ", stable Welford (mean,M2,count))";
      return std::vector<StmtPtr>{c_init_assign, scope};
    }

    // ========================= P4 — FUSED ONLINE SOFTMAX (flash) =========================
    // The softmax's TWO coupled reductions (row_max -> sub -> exp -> row_sum) stream in ONE online
    // pass carrying the running stats (m = running row-max, l = running sum-exp) with the exact
    // rescale l_new = l_old*exp(m_old - m_new) + chunk_sumexp; then a second APPLY pass re-streams R
    // substituting the finalized (M_final, L_final). x is read 2x total (A7 stream_passes=2): one DMA
    // per chunk per pass (slice_input). pin_n only (softmax reduces N) — the [free_tile,1] column-vector
    // stats broadcast over the chunk via tensor.sub -> tile.row_expand_sub. Both streamed loops are
    // stage-2 pipelines when they have at least two rolled iterations.
    if (stream_p4 && p4_kind == ::P4PatternKind::SoftmaxFlash) {
      INTERNAL_CHECK_SPAN(p4_match != nullptr && p4_match->max_stmt != nullptr &&
                              p4_match->sum_stmt != nullptr && p4_match->x_input != nullptr,
                          sp)
          << "Internal error: exact softmax P4 group has an incomplete semantic descriptor";
      const AssignStmtPtr max_stmt = p4_match->max_stmt;
      const AssignStmtPtr sum_stmt = p4_match->sum_stmt;
      const ExprPtr x_input = p4_match->x_input;
      const CallPtr max_call = As<Call>(max_stmt->value_);
      const CallPtr sum_call = As<Call>(sum_stmt->value_);

      // Emit one reduced-axis chunk's online-stats update over the [free_tile, chunk_ext] slice at
      // red_off. DMA x's slice ONCE (slice_input), then:
      //   cmax  = row_max(x);            m_new = m_in? maximum(m_in,cmax) : cmax
      //   p     = exp(sub(x, m_new));    cl    = row_sum(p)
      //   corr  = m_in? exp(sub(m_in,m_new)) : 1;   l_new = l_in? add(mul(l_in,corr), cl) : cl
      // ORDER MATTERS: compute m_new BEFORE sub/corr; l_new uses OLD l_in*corr + NEW cl. `m_in`/`l_in`
      // null => the init chunk (m=cmax, l=cl, corr=1). Returns the new ([free_tile,1]) stats.
      auto p4_chunk = [&](int64_t chunk_ext, const ExprPtr& red_off, const VarPtr& m_in, const VarPtr& l_in,
                          std::vector<StmtPtr>& out, int tag) -> std::pair<VarPtr, VarPtr> {
        const std::string tg = std::to_string(tag);
        VarPtr xs = pin_m ? slice_input(x_input, chunk_ext, free_tile, red_off, foff, out, tag)
                          : slice_input(x_input, free_tile, chunk_ext, foff, red_off, out, tag);
        auto cmax_c = reg.Create(max_call->op_->name_, {ExprPtr(xs)}, max_call->kwargs_, sp);
        auto cmax = std::make_shared<Var>(base + "_cmax" + tg, cmax_c->GetType(), sp);
        out.push_back(std::make_shared<AssignStmt>(cmax, cmax_c, sp));
        VarPtr m_new;
        if (m_in == nullptr) {
          m_new = cmax;
        } else {
          auto mn_c = reg.Create("tensor.maximum", {ExprPtr(m_in), ExprPtr(cmax)}, sp);
          m_new = std::make_shared<Var>(base + "_mnew" + tg, mn_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(m_new, mn_c, sp));
        }
        auto sh_c = reg.Create("tensor.sub", {ExprPtr(xs), ExprPtr(m_new)}, sp);  // -> row_expand_sub
        auto shv = std::make_shared<Var>(base + "_sh" + tg, sh_c->GetType(), sp);
        out.push_back(std::make_shared<AssignStmt>(shv, sh_c, sp));
        auto p_c = reg.Create("tensor.exp", {ExprPtr(shv)}, sp);
        auto pv = std::make_shared<Var>(base + "_p" + tg, p_c->GetType(), sp);
        out.push_back(std::make_shared<AssignStmt>(pv, p_c, sp));
        auto cl_c = reg.Create(sum_call->op_->name_, {ExprPtr(pv)}, sum_call->kwargs_, sp);
        auto cl = std::make_shared<Var>(base + "_cl" + tg, cl_c->GetType(), sp);
        out.push_back(std::make_shared<AssignStmt>(cl, cl_c, sp));
        VarPtr l_new;
        if (l_in == nullptr) {
          l_new = cl;
        } else {
          auto dm_c = reg.Create("tensor.sub", {ExprPtr(m_in), ExprPtr(m_new)}, sp);
          auto dm = std::make_shared<Var>(base + "_dm" + tg, dm_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(dm, dm_c, sp));
          auto corr_c = reg.Create("tensor.exp", {ExprPtr(dm)}, sp);
          auto corr = std::make_shared<Var>(base + "_corr" + tg, corr_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(corr, corr_c, sp));
          auto lc_c = reg.Create("tensor.mul", {ExprPtr(l_in), ExprPtr(corr)}, sp);
          auto lc = std::make_shared<Var>(base + "_lc" + tg, lc_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(lc, lc_c, sp));
          auto ln_c = reg.Create("tensor.add", {ExprPtr(lc), ExprPtr(cl)}, sp);
          l_new = std::make_shared<Var>(base + "_lnew" + tg, ln_c->GetType(), sp);
          out.push_back(std::make_shared<AssignStmt>(l_new, ln_c, sp));
        }
        return {m_new, l_new};
      };

      std::vector<StmtPtr> body;
      // PASS 0 — online stats. Chunk 0 inits (m=row_max(x0), l=row_sum(exp(x0-m)), corr=1); chunks
      // 1..num_full merge while threading {m_it, l_it}; the ragged tail merges after. Chunks are
      // DISJOINT (overlapping the chunk bounds themselves would double-count the reduction).
      auto [m_cur, l_cur] =
          p4_chunk(solver_stream.stats_init.extent, MakeIndex(solver_stream.stats_init.chunk_index * rc, sp),
                   nullptr, nullptr, body, 0);
      if (solver_stream.stats.trip_count > 0) {
        auto k = std::make_shared<Var>(base + "_k", index_type, sp);
        auto m_it = std::make_shared<IterArg>(base + "_m_it", m_cur->GetType(), ExprPtr(m_cur), sp);
        auto l_it = std::make_shared<IterArg>(base + "_l_it", l_cur->GetType(), ExprPtr(l_cur), sp);
        std::vector<StmtPtr> lbody;
        auto [m_new, l_new] = p4_chunk(rc, MakeMul(k, MakeIndex(rc, sp), sp), m_it, l_it, lbody, 1);
        lbody.push_back(
            std::make_shared<YieldStmt>(std::vector<ExprPtr>{ExprPtr(m_new), ExprPtr(l_new)}, sp));
        auto m_out = std::make_shared<Var>(base + "_m", m_cur->GetType(), sp);
        auto l_out = std::make_shared<Var>(base + "_l", l_cur->GetType(), sp);
        // A5: (m,l) is true loop-carried state and remains persistent. Stage=2 double-buffers only
        // the next disjoint input chunk, overlapping its load with the current chunk's online
        // reduction/merge. Chunk 0 is emitted before this loop, so require two rolled iterations.
        append_planned_loop(body, k, solver_stream.stats, std::vector<IterArgPtr>{m_it, l_it},
                            std::move(lbody), std::vector<VarPtr>{m_out, l_out});
        m_cur = m_out;
        l_cur = l_out;
      }
      if (solver_stream.stats_tail.present) {
        auto [m_new, l_new] =
            p4_chunk(solver_stream.stats_tail.extent,
                     MakeIndex(solver_stream.stats_tail.chunk_index * rc, sp), m_cur, l_cur, body, 2);
        m_cur = m_new;
        l_cur = l_new;
      }
      const VarPtr m_final = m_cur, l_final = l_cur;  // finalized [free_tile,1] stats

      // PASS 1 — apply. Re-stream R; per chunk recompute the sink cone substituting BOTH finalized
      // stats (M_final for row_max, L_final for row_sum) and assemble the [free_tile,chunk] result into
      // the full-shape [IM,IN] output. emit_strip replays sub(x,M_final)->exp->div(e,L_final) — row_max
      // and row_sum are substituted, so x is DMA'd once here too. Output threaded as an iter_arg ->
      // in-place stores (RewriteReturnedAssembleLoopToStore).
      const std::unordered_map<const Var*, VarPtr> subs = {
          {max_stmt->var_.get(), m_final}, {sum_stmt->var_.get(), l_final}};
      emit_spanning_apply(body, subs);

      auto scope = SpmdWrap(t, std::move(body), MakeIndex(num_free, sp), name, sp);
      LOG_INFO << "AutoFuse[generic]: STREAMED fused online softmax (P4) '" << name << "' ("
               << ops.size() << " ops, reduce " << (pin_m ? "M" : "N") << " ext=" << red_ext << " chunk="
               << rc << "x" << num_full << (rem ? "+tail" : "") << ", free grid " << num_free
               << ", online (m,l) stats)";
      return std::vector<StmtPtr>{c_init_assign, scope};
    }

    std::vector<StmtPtr> body;
    // PASS 0 — accumulate the reduction over disjoint reduced-axis chunks (stop at the reduction op).
    // The accumulator is the small reduced [.,1]/[1,.]; chunk 0 inits it, chunks 1.. merge in a
    // carried loop (acc iter_arg, spike-proven), the ragged tail merges after.
    std::unordered_map<const Var*, VarPtr> oc0;
    VarPtr acc =
        strip_at(solver_stream.stats_init.extent, MakeIndex(solver_stream.stats_init.chunk_index * rc, sp),
                 body, oc0, red_stmt.get(), nullptr);
    if (solver_stream.stats.trip_count > 0) {
      auto k = std::make_shared<Var>(base + "_k", index_type, sp);
      auto acc_it = std::make_shared<IterArg>(base + "_acc_it", acc->GetType(), ExprPtr(acc), sp);
      std::vector<StmtPtr> lbody;
      std::unordered_map<const Var*, VarPtr> ock;
      VarPtr part = strip_at(rc, MakeMul(k, MakeIndex(rc, sp), sp), lbody, ock, red_stmt.get(), nullptr);
      auto m_call = reg.Create(merge_op, {ExprPtr(acc_it), ExprPtr(part)}, sp);
      auto acc_n = std::make_shared<Var>(base + "_acc_n", m_call->GetType(), sp);
      lbody.push_back(std::make_shared<AssignStmt>(acc_n, m_call, sp));
      lbody.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{acc_n}, sp));
      auto acc_out = std::make_shared<Var>(base + "_acc", acc->GetType(), sp);
      // A5 (G2): PIPELINE the accumulate pass. The running accumulator `acc_it` is a TRUE loop carry
      // (acc_n = merge(acc_it, part_k)), so it stays single-buffered/persistent; only the per-chunk
      // load+reduce (`part`) double-buffers — stage=2 overlaps chunk k+1's load with chunk k's
      // reduce+merge, hiding the DDR-bound input read behind compute (this is the P1/P2 FIRST pass).
      // Pipeline only when the trip (num_full-1) is >= 2; else nothing to overlap.
      append_planned_loop(body, k, solver_stream.stats, std::vector<IterArgPtr>{acc_it}, std::move(lbody),
                          std::vector<VarPtr>{acc_out});
      acc = acc_out;
    }
    if (solver_stream.stats_tail.present) {
      std::unordered_map<const Var*, VarPtr> oct;
      VarPtr tpart =
          strip_at(solver_stream.stats_tail.extent, MakeIndex(solver_stream.stats_tail.chunk_index * rc, sp),
                   body, oct, red_stmt.get(), nullptr);
      auto m_call = reg.Create(merge_op, {ExprPtr(acc), ExprPtr(tpart)}, sp);
      auto acc_t = std::make_shared<Var>(base + "_acc_t", m_call->GetType(), sp);
      body.push_back(std::make_shared<AssignStmt>(acc_t, m_call, sp));
      acc = acc_t;
    }

    if (stream_p1) {
      // P1: every live-out remains folded along the reduced axis. A bare reduction assembles the
      // accumulator directly. If thin pointwise finalization follows the reduction (for example
      // row_sum -> muls), replay that reduced-shape cone ONCE with the finalized accumulator
      // substituted; it does not require P2's full reduced-axis apply re-stream.
      VarPtr folded_out = acc;
      if (!sink_is_reduction) {
        const std::unordered_map<const Var*, VarPtr> subs = {{red_stmt->var_.get(), acc}};
        std::unordered_map<const Var*, VarPtr> oc_folded;
        folded_out = strip_at(1, MakeIndex(0, sp), body, oc_folded, nullptr, &subs);
      }
      ExprPtr asm_off = pin_m ? MakeTuple2(MakeIndex(0, sp), foff, sp)   // [1,w] at [0, n]
                              : MakeTuple2(foff, MakeIndex(0, sp), sp);  // [h,1] at [m, 0]
      auto asm_call =
          reg.Create("tensor.assemble", {ExprPtr(c_init), ExprPtr(folded_out), asm_off}, sp);
      body.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));
    } else {
      // P2 PASS 1 — final apply. The output spans the reduced axis, so re-stream it: for each chunk,
      // recompute the sink dependency cone (subs the finalized reduction `acc`) over the [free_tile,
      // chunk] slice and assemble it into the full-shape sink at the chunk's reduced-axis offset.
      // Output threaded as an iter_arg (assemble-into-output -> in-place stores, like the pipeline).
      const std::unordered_map<const Var*, VarPtr> subs = {{red_stmt->var_.get(), acc}};
      emit_spanning_apply(body, subs);
    }

    auto scope = SpmdWrap(t, std::move(body), MakeIndex(num_free, sp), name, sp);
    LOG_INFO << "AutoFuse[generic]: STREAMED " << (stream_p2 ? "reduction+apply (P2)" : "reduction (P1)")
             << " '" << name << "' (" << ops.size() << " ops, " << (pin_m ? "reduce M" : "reduce N")
             << " ext=" << red_ext << " chunk=" << rc << "x" << num_full << (rem ? "+tail" : "")
             << ", free grid " << num_free << ", " << merge_op << ")";
    return std::vector<StmtPtr>{c_init_assign, scope};
  }

  // MULTI-SINK: a group with >1 live-out (the solver merges sinks that share inputs).
  // Serial for now (pipeline/split are the single-sink refinements): replay the group ONCE
  // in the solver's execution order, then assemble EACH sink into its own output buffer at
  // its (projected) offset. The shared inputs stay resident across sinks precisely because
  // the replay follows the pebbling order — that is the whole point of the merge.
  if (sinks.size() > 1) {
    std::vector<StmtPtr> prologue;  // one tensor.create per sink
    std::vector<StmtPtr> mbody;
    std::unordered_map<const Var*, VarPtr> onchip;
    emit_strip(h, w, mi, ni, mbody, onchip);  // replay all ops; onchip[var] = every op's tile
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
  // is 16-aligned (PTO Fusebox ascend910b_cost.cpp:886, mirroring the matmul kfrac gate id.:870), so
  // IM/S is 16-aligned and this gate holds. The gate is kept as DEFENSE-IN-DEPTH: a non-conforming
  // S from any other cost model declines to the CORRECT non-split body rather than emitting an OOB
  // read. Everything else (max/min, row-reduction split, reduction-feeds-pointwise) also declines.
  // Dependency-cone safety: S2 replays the WHOLE group over each disjoint M-slice, so a per-slice
  // result is a valid atomic-add partial ONLY when every op UPSTREAM of the terminal col_sum is
  // pointwise/broadcast. If another op reduces M (a prior col_max / col_sum / col_min), its
  // per-slice result is a LOCAL reduction, not the global one -> wrong (col_max->sub->col_sum would
  // subtract each slice's own max). Decline S2 for such a cone. The solver currently groups a prior
  // M-reduction into a SEPARATE group (verified), so this is defense-in-depth — correct regardless
  // of the fusion decision.
  bool cone_reduces_m_upstream = false;
  for (const auto& a : ops) {
    if (a == out_stmt) continue;
    auto c = As<Call>(a->value_);
    if (c == nullptr || ClassifyOp(c) != ::OpType::Reduction || c->args_.empty()) continue;
    const auto [ciM, ciN] = Static2DShape(c->args_[0]->GetType());
    const auto [coM, coN] = Static2DShape(a->var_->GetType());
    if (ciM > 1 && coM == 1) cone_reduces_m_upstream = true;  // an upstream col (M) reduction
  }
  const bool has_split_plan =
      solver_stream.reduction_split_kind == ::VectorReductionSplitKind::ColSumAtomicAdd;
  if (has_split_plan) {
    // The solver-owned plan is the admission decision. Re-check only the IR and
    // arithmetic invariants needed to build it; a mismatch is a Tier-B contract
    // failure, never a reason to pair an S>1 cost with a serial kernel.
    auto split_sink = As<Call>(out_stmt->value_);
    const int64_t S = solver_stream.reduction_split_factor;
    const int64_t rsz = solver_stream.reduction_partial_extent;
    const ::VectorReductionSeedPlan& seed = solver_stream.reduction_seed;
    if (!pin_m || cone_reduces_m_upstream || split_sink == nullptr ||
        !IsOp(split_sink, "tensor.col_sum") || S != tile.split || S <= 1 ||
        rsz <= 0 || rsz * S != IM || rsz % g != 0 || IN % w != 0 ||
        !seed.present || seed.work_units != num_n || seed.valid_rows != M ||
        seed.valid_cols != w) {
      return GenericDeclineB(
          "solver col_sum atomic split descriptor disagrees with emitted IR/geometry (G6/S2)",
          out_stmt->span_);
    }
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
    VarPtr part = emit_strip(rsz, w, r_mi, sni, sbody, oc_split);  // [1,w] partial col_sum over the M-slice
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_seeded), part, MakeTuple2(MakeIndex(0, sp), sni, sp)},
                               {{"atomic", 1}}, sp);
    sbody.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));
    auto scope = SpmdWrap(t2, std::move(sbody), MakeIndex(num_n * S, sp), name, sp);
    LOG_INFO << "AutoFuse[generic]: SUM col-reduction group '" << name << "' split " << S
             << " ways over the reduced axis (S2, atomic-add merge)";
    return std::vector<StmtPtr>{c_init_assign, seed_scope, scope};
  }
  // A selected S>1 vector plan without the exact solver descriptor is illegal.
  // Production falls back safely; strict mode makes the model/emit regression
  // loud. The normal solver no longer enumerates such a candidate.
  if (tile.split > 1) {
    return GenericDeclineB(
        "vector reduction split has no realizable solver descriptor (G6/S2)",
        out_stmt->span_);
  }

  // G1 (emit defense): a MULTI-reduction group (softmax/layernorm: >1 reduction over the reduced
  // axis) that reached the materialized path pins its reduced axis FULL. If even the thinnest tile
  // — one free lane over the full reduced axis, one band per op — overflows UB, no materialized
  // tiling fits and the emit cannot stream it (streaming handles a SINGLE reduction only; the
  // online multi-reduction path P4 is not built). The BUILDABLE cost model marks such groups
  // infeasible so the solver cuts them (an unfused softmax IS buildable); this is defense-in-depth
  // for an analytic plan that slips through — decline to the legacy tiler rather than emit an
  // over-UB tile that fails downstream at AllocateMemoryAddr. Small multi-reduction groups (the
  // thinnest tile fits) and single-reduction streamed groups (returned above) are unaffected.
  if (has_reduction && p1_nreds > 1 && (pin_m || pin_n) && !stream_p4) {
    const int64_t g1_red = pin_m ? IM : IN;  // pinned reduced axis (full)
    if (static_cast<int64_t>(ops.size()) * g1_red * p1_dtb > p1_ub)
      return GenericDeclineB("streamed multi-reduction (softmax/layernorm) over a reduced axis too "
                             "large for UB — the online multi-reduction path (P4) is not yet built; "
                             "declining to the legacy tiler", out_stmt->span_);
  }

  // Materialized and pointwise strip scheduling is solver-owned.  The plan was
  // derived while costing this exact configuration from the same DFS liveness
  // order; emission only validates and realizes it.
  const ::VectorStreamKind expected_body_kind =
      solver_stream.streamed() ? ::VectorStreamKind::Pointwise : ::VectorStreamKind::Materialized;
  const bool body_plan_matches =
      solver_stream.feasible && solver_stream.kind == expected_body_kind && solver_stream.tile_h == h &&
      solver_stream.tile_w == w && solver_stream.row_strips >= 1 && solver_stream.width_strips >= 1 &&
      solver_stream.work_units == num_m * num_n &&
      solver_stream.strip_h == (h + solver_stream.row_strips - 1) / solver_stream.row_strips &&
      solver_stream.strip_w > 0 && solver_stream.strip_w <= AlignUp(w, g) &&
      solver_stream.body.first_chunk == 0 &&
      solver_stream.body.trip_count == solver_stream.row_strips * solver_stream.width_strips;
  INTERNAL_CHECK_SPAN(body_plan_matches, sp)
      << "Internal error: materialized/pointwise emit does not match solver vector plan for group '" << name
      << "'";
  const int64_t num_strips = solver_stream.row_strips;
  const int64_t num_wstrips = solver_stream.width_strips;
  const int64_t strip_w = solver_stream.strip_w;

  std::vector<StmtPtr> body_stmts;
  if (num_strips < 2 && num_wstrips < 2) {
    // Serial (matches the cost model's db=false): the whole tile is one strip. Only reached when the
    // whole [h,w] tile fits UB — an over-UB tile bumped num_strips/num_wstrips >= 2 above.
    std::unordered_map<const Var*, VarPtr> oc_serial;
    VarPtr tv = emit_strip(h, w, mi, ni, body_stmts, oc_serial);
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(c_init), tv, MakeTuple2(mi, ni, sp)}, sp);
    body_stmts.push_back(std::make_shared<AssignStmt>(c_var, asm_call, sp));
  } else {
    // Flatten the solver-owned row strips x width strips into one carried loop.
    // The plan chooses Sequential when the strip is below the double-buffer
    // floor, otherwise Pipeline(stage=2).
    //   strip at (mi + srow, ni + scol); replay ops -> [strip_h, emit_w] tile; out = assemble(out,
    //   strip, off); yield.
    // num_wstrips == 1 is the common ROW-only case: srow = s*strip_h over the full width w, byte-
    // identical to the pre-C2 emit (sni == ni, emit_w == w). num_wstrips >= 2 (C2) additionally chunks
    // the contiguous width so a too-wide tile still streams. Strips are UNIFORM (ceil height/width); the
    // ragged last row/col strip's offset is clamped in-bounds — an overlapping re-compute, idempotent
    // for the non-atomic pointwise assemble (same clamp trick as the SPMD grid mi/ni and the reduction
    // free axis). The output iter_arg is used ONLY as the assemble target (source/offset reference s,
    // not the iter_arg), so ConvertTensorToTileOps::RewriteReturnedAssembleLoopToStore lowers it to an
    // in-place tile.store while preserving kind==Pipeline + pipeline_stages.
    const int64_t strip_h = (h + num_strips - 1) / num_strips;  // ceil, uniform
    const bool strip_ragged = strip_h * num_strips > h;         // last row strip overruns h -> clamp
    const bool wstrip_ragged = strip_w * num_wstrips > w;       // last width strip overruns w -> clamp
    const int64_t total_strips = num_strips * num_wstrips;
    auto index_ty = std::make_shared<ScalarType>(DataType::INDEX);
    auto s = std::make_shared<Var>(base + "_s", index_ty, sp);
    auto out_iter = std::make_shared<IterArg>(base + "_out_it", c_init->GetType(), ExprPtr(c_init), sp);
    // Decode the flat strip index into (row strip, width strip). num_wstrips == 1 keeps the row-only
    // decode (srow_idx == s, full width) so the pre-C2 emit is reproduced exactly.
    ExprPtr srow_idx = s;
    if (num_wstrips > 1) srow_idx = MakeFloorDiv(s, MakeIndex(num_wstrips, sp), sp);
    ExprPtr srow = MakeMul(srow_idx, MakeIndex(strip_h, sp), sp);
    if (strip_ragged) srow = MakeMin(srow, MakeIndex(h - strip_h, sp), sp);
    ExprPtr smi = MakeAdd(mi, srow, sp);
    ExprPtr sni = ni;
    int64_t emit_w = w;
    if (num_wstrips > 1) {
      ExprPtr scol = MakeMul(MakeFloorMod(s, MakeIndex(num_wstrips, sp), sp), MakeIndex(strip_w, sp), sp);
      if (wstrip_ragged) scol = MakeMin(scol, MakeIndex(w - strip_w, sp), sp);
      sni = MakeAdd(ni, scol, sp);
      emit_w = strip_w;
    }
    std::vector<StmtPtr> loop_body;
    std::unordered_map<const Var*, VarPtr> oc_pipe;
    VarPtr tv = emit_strip(strip_h, emit_w, smi, sni, loop_body, oc_pipe);
    auto asm_call = reg.Create("tensor.assemble", {ExprPtr(out_iter), tv, MakeTuple2(smi, sni, sp)}, sp);
    auto out_next = std::make_shared<Var>(base + "_out_n", asm_call->GetType(), sp);
    loop_body.push_back(std::make_shared<AssignStmt>(out_next, asm_call, sp));
    loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{out_next}, sp));
    StmtPtr body = SeqStmts::Flatten(std::move(loop_body), sp);
    const bool pipelined = solver_stream.body.pipeline_stages == 2;
    std::vector<std::pair<std::string, std::any>> loop_attrs;
    if (pipelined) {
      loop_attrs.emplace_back(kPipelineStagesAttr, /*stages=*/2);
    }
    auto for_stmt =
        std::make_shared<ForStmt>(s, MakeIndex(0, sp), MakeIndex(total_strips, sp), MakeIndex(1, sp),
                                  std::vector<IterArgPtr>{out_iter}, body, std::vector<VarPtr>{c_var}, sp,
                                  pipelined ? ForKind::Pipeline : ForKind::Sequential, std::move(loop_attrs));
    body_stmts.push_back(for_stmt);
  }

  auto scope = SpmdWrap(t, std::move(body_stmts), MakeIndex(solver_stream.work_units, sp), name, sp);
  LOG_INFO << "AutoFuse[generic]: " << (has_reduction ? "elementwise+reduction" : "elementwise") << " group '"
           << name << "' tiled by the generic driver (" << ops.size() << " ops, grid " << num_m << "x"
           << num_n << " = " << solver_stream.work_units << " logical regions, "
           << (num_strips * num_wstrips)
           << " planned strips (stage=" << solver_stream.body.pipeline_stages << ")"
           << (num_wstrips > 1 ? " [width-chunked]" : "") << ")";
  return std::vector<StmtPtr>{c_init_assign, scope};
}

// The MATMUL rule (increment 3), for a LONE matmul (chains are S4 -> fall back). Its
// tiling IS TileMatmul (the grid + split-K tiled-seed + BuildTileMatmul k-pipeline); the
// generic path adds the fail-loud A3/SR7 contract asserts the legacy tiler lacks and
// unifies dispatch behind the flag. The tiling body is REUSED for now; absorbing it so
// the legacy TileMatmul dispatch can retire is the parity-prep step.
std::optional<std::vector<StmtPtr>> EmitLoneMatmulGeneric(const AssignStmtPtr& assign,
                                                          const SolverTile& tile,
                                                          const std::string& name) {
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

  if (tile.cube_schedule.feasible && tile.cube_schedule.matmuls.size() == 1) {
    const auto& loop = tile.cube_schedule.matmuls.front().k_loop;
    if ((tile.cube_schedule.split_k > 1 || loop.full_chunks >= 2) &&
        !CubeOutputCanCarryKLoop(call->args_[0], ct->dtype_)) {
      return GenericDeclineB("planned K carry requires FP32/INT32 accumulation before final output narrowing",
                             assign->span_);
    }
  }

  // Non-uniform grid is now REALIZED (G-A): TileMatmul's non-split path tiles a
  // ceil(M/h) x ceil(N/w) grid with clamped (overlapping, idempotent) offsets, so a
  // parts_m/parts_n grid whose max-extent w/h does not divide the output is faithfully
  // emitted rather than declined. The split-K path stays divisor-only (its atomic-add
  // merge cannot tolerate the clamp overlap); TileMatmul declines that case internally.
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
// static-shape chain or the tile does not divide the output. When the per-tile
// T_band [h,K2] exceeds L0c, DEEP-T (G-B) tiles the shared K2 into panels so MM2
// becomes a matmul_acc chain and only [h,k2p] is on-chip at a time (declines if no
// 16-aligned divisor panel fits L0c). TODO: share the wrapper with TileMatmul.
std::optional<std::vector<StmtPtr>> TileChainedMatmul(const AssignStmtPtr& mm1,
                                                      const AssignStmtPtr& mm2,
                                                      const SolverTile& tile,
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

  // Deep-T (G-B): the per-tile intermediate T_band [h,K2] (MM1's output) must fit L0c.
  // When it exceeds L0c, tile the SHARED dimension K2 into panels of `k2p` (each T_panel
  // [h,k2p] fits L0c): MM1 computes T_panel = A[mi:mi+h,:] @ B[:, k2o:k2o+k2p]; MM2
  // ACCUMULATES T_panel @ D[k2o:k2o+k2p, ni:ni+w] into the output tile via matmul_acc, so
  // T never fully materializes (only [h,k2p] is on-chip at a time). k2p is the largest
  // 16-aligned DIVISOR of K2 that fits L0c; if none fits, decline (the whole-output fused
  // scope still computes correctly with T on-chip — the current fallback).
  const int64_t dtb = std::max<int64_t>(1, static_cast<int64_t>(dtype.GetBit()) / 8);
  const int64_t l0c = ReadHwParams().cube_capacity;
  const bool deep_t = (h * K2 * dtb > l0c);
  int64_t k2p = K2;
  if (deep_t) {
    const int64_t max_p = l0c / std::max<int64_t>(1, h * dtb);  // widest panel fitting L0c
    k2p = 0;
    for (int64_t p = (max_p / 16) * 16; p >= 16; p -= 16)
      if (K2 % p == 0) { k2p = p; break; }
    if (k2p == 0) return std::nullopt;  // no 16-aligned divisor panel fits L0c -> flush handles it
  }
  const std::vector<std::pair<std::string, std::any>> mm2_kw = {
      {"a_trans", false}, {"b_trans", false}, {"c_matrix_nz", false}, {"out_dtype", dtype}};
  const std::vector<std::pair<std::string, std::any>> acc2_kw = {{"a_trans", false}, {"b_trans", false}};

  // Inner serial chain for one output tile at element offset [mi,ni]:
  //   T_band = A[mi:mi+h, :] @ B            -> [h,K2]  (k-pipelined: A streams from DDR)
  //   out_tile = T_band @ D[:, ni:ni+w]     -> [h,w]   (single: T_band is on-chip)
  // Deep-T variant (T_band exceeds L0c): panel over K2 so MM2 is a matmul_acc chain.
  auto build_chain = [&](const ExprPtr& mi, const ExprPtr& ni, const VarPtr& out_tile) {
    std::vector<StmtPtr> stmts;
    if (!deep_t) {
      auto tband_type =
          std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(K2, sp)}, dtype);
      auto tband = std::make_shared<Var>(base + "_tband", tband_type, sp);
      auto s1 = BuildTileMatmul(A, B, mi, MakeIndex(0, sp), h, K2, K1, tile.k, dtype, tband, base + "_t", sp);
      auto s2 = BuildTileMatmul(tband, D, MakeIndex(0, sp), ni, h, w, K2, /*k=*/0, dtype, out_tile, base + "_c", sp);
      for (auto& s : s1) stmts.push_back(std::move(s));
      for (auto& s : s2) stmts.push_back(std::move(s));
      return stmts;
    }
    // Deep-T: for each K2-panel [k2o, k2o+k2p), compute T_panel = A[mi:mi+h,:] @ B[:, panel]
    // (k-pipelined over K1) and fold T_panel @ D[panel, ni:ni+w] into the output accumulator.
    // The last panel binds out_tile; the first MM2 is a plain matmul, the rest matmul_acc.
    const int64_t num_p = K2 / k2p;
    auto out_ty = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, dtype);
    VarPtr acc_var;
    for (int64_t p = 0; p < num_p; ++p) {
      const std::string pb = base + "_p" + std::to_string(p);
      const ExprPtr k2o = MakeIndex(p * k2p, sp);
      // MM1: T_panel[h,k2p] = A[mi:mi+h, :] @ B[:, k2o:k2o+k2p]  (BuildTileMatmul's ni=k2o, w=k2p).
      auto tp_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(k2p, sp)}, dtype);
      auto tpanel = std::make_shared<Var>(pb + "_tp", tp_type, sp);
      for (auto& s : BuildTileMatmul(A, B, mi, k2o, h, k2p, K1, tile.k, dtype, tpanel, pb + "_t", sp))
        stmts.push_back(std::move(s));
      // D panel: D[k2o:k2o+k2p, ni:ni+w] -> [k2p, w].
      auto dslice = reg.Create("tensor.slice", {D, MakeIndexTuple({k2p, w}, sp), MakeTuple2(k2o, ni, sp)}, sp);
      auto dpv = std::make_shared<Var>(pb + "_d", dslice->GetType(), sp);
      stmts.push_back(std::make_shared<AssignStmt>(dpv, dslice, sp));
      // MM2: p==0 -> matmul; else matmul_acc(acc, T_panel, D_panel). Last panel binds out_tile.
      const VarPtr res = (p == num_p - 1) ? out_tile : std::make_shared<Var>(pb + "_ac", out_ty, sp);
      ExprPtr mm2c = (p == 0)
                         ? reg.Create("tensor.matmul", {ExprPtr(tpanel), ExprPtr(dpv)}, mm2_kw, sp)
                         : reg.Create("tensor.matmul_acc", {ExprPtr(acc_var), ExprPtr(tpanel), ExprPtr(dpv)}, acc2_kw, sp);
      stmts.push_back(std::make_shared<AssignStmt>(res, mm2c, sp));
      acc_var = res;
    }
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

ExprPtr ScaleIndex(const ExprPtr& index, int64_t extent, const Span& sp) {
  if (auto value = As<ConstInt>(index); value != nullptr && value->value_ == 0) {
    return MakeIndex(0, sp);
  }
  if (extent == 1) return index;
  return MakeMul(index, MakeIndex(extent, sp), sp);
}

ExprPtr CubeRegionAxisOffset(::CubeAxisBinding binding, int64_t extent, const ExprPtr& m_index,
                             const ExprPtr& n_index, const ExprPtr& split_index, const Span& sp) {
  switch (binding) {
    case ::CubeAxisBinding::Full:
    case ::CubeAxisBinding::SequentialK:
      return MakeIndex(0, sp);
    case ::CubeAxisBinding::SpatialM:
      return ScaleIndex(m_index, extent, sp);
    case ::CubeAxisBinding::SpatialN:
      return ScaleIndex(n_index, extent, sp);
    case ::CubeAxisBinding::ParallelK:
      return ScaleIndex(split_index, extent, sp);
  }
  INTERNAL_UNREACHABLE << "Internal error: unknown CubeAxisBinding";
}

struct ValidatedCubeGroup {
  std::unordered_map<size_t, AssignStmtPtr> op_assigns;
  int64_t parts_m = 0;
  int64_t parts_n = 0;
  int64_t region_h = 0;
  int64_t region_w = 0;
};

std::optional<ValidatedCubeGroup> ValidateCubeScheduleGroup(
    const std::vector<StmtPtr>& members, const std::unordered_map<const Stmt*, size_t>& stmt_op,
    const SolverTile& tile, std::string* reason) {
  const ::CubeSchedulePlan& plan = tile.cube_schedule;
  auto fail = [&](std::string message) {
    *reason = std::move(message);
    return std::optional<ValidatedCubeGroup>{};
  };
  if (!plan.feasible || !plan.emit_compatible || plan.matmuls.empty()) {
    return fail("missing feasible emit-compatible CubeSchedulePlan");
  }
  if (plan.model_overlap_granted && !plan.overlap_implementable) {
    return fail("cost grants GM/compute overlap but a planned K loop is serial");
  }

  ValidatedCubeGroup validated;
  for (const StmtPtr& stmt : members) {
    auto op_it = stmt_op.find(stmt.get());
    auto assign = As<AssignStmt>(stmt);
    auto call = assign != nullptr ? As<Call>(assign->value_) : nullptr;
    if (op_it == stmt_op.end() || assign == nullptr || call == nullptr || !IsOp(call, "tensor.matmul") ||
        call->args_.size() != 2) {
      return fail("group contains a statement that is not a registered binary matmul");
    }
    validated.op_assigns.emplace(op_it->second, assign);
  }

  validated.parts_m = plan.m_partition.parts;
  validated.parts_n = plan.n_partition.parts;
  validated.region_h = plan.m_partition.big;
  validated.region_w = plan.n_partition.big;
  if (validated.parts_m <= 0 || validated.parts_n <= 0 || validated.region_h <= 0 ||
      validated.region_w <= 0 || plan.m_partition.num_big != 0 || plan.n_partition.num_big != 0 ||
      plan.spatial_tiles != validated.parts_m * validated.parts_n || plan.split_k != tile.split ||
      plan.seed_required != (plan.split_k > 1) || plan.seed.present != plan.seed_required ||
      (plan.seed.present && (plan.seed.work_units <= 0 || plan.seed.valid_rows <= 0 ||
                             plan.seed.valid_cols != validated.region_w || plan.seed.bytes <= 0)) ||
      plan.work_units != plan.spatial_tiles * plan.split_k) {
    return fail("non-uniform or inconsistent spatial/split geometry");
  }

  size_t root_count = 0;
  int64_t root_m = -1;
  int64_t root_n = -1;
  std::set<std::tuple<size_t, int, int, int>> boundary_requests;
  for (size_t instance = 0; instance < plan.matmuls.size(); ++instance) {
    if (instance > static_cast<size_t>(std::numeric_limits<int64_t>::max())) {
      return fail("cube request-instance index exceeds the signed dependency range");
    }
    const int64_t instance_id = static_cast<int64_t>(instance);
    const ::CubeMatmulSchedule& mm = plan.matmuls[instance];
    auto assign_it = validated.op_assigns.find(mm.op);
    const bool lhs_dependency_invalid = mm.lhs_producer >= instance_id;
    const bool rhs_dependency_invalid = mm.rhs_producer >= instance_id;
    if (mm.instance != instance || assign_it == validated.op_assigns.end() || lhs_dependency_invalid ||
        rhs_dependency_invalid || mm.lhs_ephemeral != (mm.lhs_producer >= 0) ||
        mm.rhs_ephemeral != (mm.rhs_producer >= 0)) {
      return fail("request-instance identity or dependency invariant is invalid");
    }
    auto call = As<Call>(assign_it->second->value_);
    auto output_type = As<TensorType>(assign_it->second->var_->GetType());
    if (output_type == nullptr) return fail("matmul output is not a TensorType");
    auto lhs_type = AsTensorTypeLike(call->args_[0]->GetType());
    if (lhs_type == nullptr ||
        mm.accumulator_dtype != MapSolverDType(CubeAccumulatorDType(lhs_type->dtype_)) ||
        mm.storage_dtype != MapSolverDType(output_type->dtype_)) {
      return fail("matmul accumulator/storage dtype descriptor does not match the source IR");
    }
    const auto [output_m, output_n] = Static2DShape(assign_it->second->var_->GetType());
    const auto [lhs_m, lhs_k] = Static2DShape(call->args_[0]->GetType());
    const auto [rhs_k, rhs_n] = Static2DShape(call->args_[1]->GetType());
    if (output_m <= 0 || output_n <= 0 || lhs_m != output_m || rhs_n != output_n || lhs_k <= 0 ||
        rhs_k != lhs_k || mm.contraction != lhs_k || mm.output.height <= 0 || mm.output.width <= 0 ||
        mm.k_loop.chunk <= 0 || mm.k_loop.chunk % 16 != 0 || mm.effective_contraction <= 0 ||
        mm.effective_contraction % 16 != 0 ||
        mm.k_loop.full_chunks != mm.effective_contraction / mm.k_loop.chunk ||
        mm.k_loop.tail != mm.effective_contraction - mm.k_loop.full_chunks * mm.k_loop.chunk ||
        mm.k_loop.tail % 16 != 0) {
      return fail("matmul shape or K-loop descriptor does not match the source IR");
    }
    if (mm.k_loop.pipeline_stages >= 2 && mm.k_loop.full_chunks < 3) {
      return fail("stage-2 K loop has no two-iteration rolled steady state");
    }
    const bool has_chunked_carry = mm.k_loop.full_chunks >= 2;
    if (mm.output_tile_m <= 0 || mm.output_tile_n <= 0 ||
        mm.output_tiles_m != CeilDiv(mm.output.height, mm.output_tile_m) ||
        mm.output_tiles_n != CeilDiv(mm.output.width, mm.output_tile_n) || mm.output_variants.empty()) {
      return fail("output/L0C tile grid is missing or inconsistent");
    }
    int64_t variant_tiles = 0;
    for (const ::CubeOutputTileVariant& variant : mm.output_variants) {
      const int64_t init_k = has_chunked_carry ? mm.k_loop.chunk : mm.effective_contraction;
      auto l0_loop_matches = [](const ::L0MatmulPlan& child, int64_t extent) {
        return child.k_loop.chunk == child.k && child.k_loop.full_chunks >= 0 && child.k_loop.tail >= 0 &&
               child.k_loop.full_chunks * child.k_loop.chunk + child.k_loop.tail == extent &&
               child.k_loop.pipeline_stages == (child.k_loop.full_chunks >= 2 ? 2 : 1) &&
               child.phases.wall_cycles == child.phases.init_cycles + child.phases.rolled_cycles +
                                               child.phases.tail_cycles + child.phases.drain_cycles;
      };
      variant_tiles += variant.count;
      if (variant.height <= 0 || variant.width <= 0 || variant.count <= 0 || !variant.l0_init.feasible ||
          variant.l0_init.m != variant.height || variant.l0_init.n != variant.width ||
          variant.l0_init.output_target != ::L0OutputTarget::Acc ||
          !l0_loop_matches(variant.l0_init, init_k) || variant.l0_rolled.feasible != has_chunked_carry ||
          (variant.l0_rolled.feasible &&
           (variant.l0_rolled.m != variant.height || variant.l0_rolled.n != variant.width ||
            variant.l0_rolled.output_target != ::L0OutputTarget::Acc ||
            !l0_loop_matches(variant.l0_rolled, mm.k_loop.chunk))) ||
          variant.l0_tail.feasible != (mm.k_loop.tail > 0) ||
          (variant.l0_tail.feasible &&
           (variant.l0_tail.m != variant.height || variant.l0_tail.n != variant.width ||
            variant.l0_tail.output_target != ::L0OutputTarget::Acc ||
            !l0_loop_matches(variant.l0_tail, mm.k_loop.tail)))) {
        return fail("shared-backend L0 child phase does not keep one complete output tile in L0C");
      }
    }
    const int64_t expected_drain_bytes =
        mm.output.height * mm.output.width *
        std::max<int64_t>(1, static_cast<int64_t>(output_type->dtype_.GetBit()) / 8);
    if (!mm.final_drain.required || mm.final_drain.target_l1 != !mm.is_sink ||
        mm.final_drain.atomic != (mm.is_sink && plan.split_k > 1) ||
        mm.final_drain.valid_rows != mm.output_tile_m || mm.final_drain.valid_cols != mm.output_tile_n ||
        mm.final_drain.tile_count != variant_tiles ||
        variant_tiles != mm.output_tiles_m * mm.output_tiles_n ||
        mm.final_drain.bytes != expected_drain_bytes || mm.final_drain.cycles <= 0.0) {
      return fail("post-K-loop Acc drain descriptor is inconsistent");
    }
    if (mm.final_drain.target_l1 && output_type->dtype_ != DataType::BF16 &&
        output_type->dtype_ != DataType::FP16) {
      return fail("A2/A3 internal Acc-to-Mat drain requires BF16/FP16 storage");
    }
    if (mm.is_sink) {
      ++root_count;
      if (mm.output.height_binding != ::CubeAxisBinding::SpatialM ||
          mm.output.width_binding != ::CubeAxisBinding::SpatialN || mm.output.height != validated.region_h ||
          mm.output.width != validated.region_w || validated.parts_m * validated.region_h != output_m ||
          validated.parts_n * validated.region_w != output_n) {
        return fail("root output does not exactly match the uniform spatial partition");
      }
      if (root_m < 0) {
        root_m = output_m;
        root_n = output_n;
      } else if (root_m != output_m || root_n != output_n) {
        return fail("cube roots do not share one output grid");
      }
    }

    auto record_boundary = [&](const ::CubeTensorRegionPlan& region, int port, int64_t producer) {
      if (producer >= 0) return true;
      return boundary_requests
          .emplace(region.tensor, port, static_cast<int>(region.height_binding),
                   static_cast<int>(region.width_binding))
          .second;
    };
    if (!record_boundary(mm.lhs, 0, mm.lhs_producer) || !record_boundary(mm.rhs, 1, mm.rhs_producer)) {
      return fail("cost-deduplicated boundary request needs an explicit shared L1 lifetime");
    }
  }
  if (root_count == 0 || (plan.split_k > 1 && root_count != 1)) {
    return fail("parallel split requires exactly one boundary root");
  }
  return validated;
}

struct CubeSplitSeed {
  StmtPtr scope;
  ExprPtr output;
};

CubeSplitSeed BuildCubeSplitSeed(const AssignStmtPtr& root_assign, const ExprPtr& initial_output,
                                 const ::CubeSplitSeedPlan& seed, int64_t parts_n, const std::string& name,
                                 const Span& sp) {
  auto& registry = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
  const auto [output_m, output_n] = Static2DShape(root_assign->var_->GetType());
  auto output_type = As<TensorType>(root_assign->var_->GetType());
  const DataType dtype = output_type->dtype_;
  const int64_t dtype_bytes = std::max<int64_t>(1, static_cast<int64_t>(dtype.GetBit()) / 8);
  const int64_t seed_h = seed.valid_rows;
  const int64_t region_w = seed.valid_cols;
  const int64_t seed_m_parts = CeilDiv(output_m, seed_h);
  INTERNAL_CHECK(seed.present && seed.work_units == seed_m_parts * parts_n &&
                 seed.bytes == seed.work_units * seed_h * region_w * dtype_bytes)
      << "AutoFuse cube split seed does not match CubeSchedulePlan";
  auto seed_index = std::make_shared<Var>(root_assign->var_->name_hint_ + "_seed_index", index_type, sp);
  ExprPtr seed_m = ScaleIndex(MakeFloorDiv(seed_index, MakeIndex(parts_n, sp), sp), seed_h, sp);
  if (seed_m_parts * seed_h > output_m) {
    seed_m = MakeMin(seed_m, MakeIndex(output_m - seed_h, sp), sp);
  }
  auto seed_n = ScaleIndex(MakeFloorMod(seed_index, MakeIndex(parts_n, sp), sp), region_w, sp);
  auto zero = std::make_shared<ConstFloat>(0.0, dtype, sp);
  auto fill =
      registry.Create("tensor.full", {MakeIndexTuple({seed_h, region_w}, sp), zero}, {{"dtype", dtype}}, sp);
  auto fill_var = std::make_shared<Var>(root_assign->var_->name_hint_ + "_seed_zero", fill->GetType(), sp);
  auto assemble =
      registry.Create("tensor.assemble", {initial_output, fill_var, MakeTuple2(seed_m, seed_n, sp)}, sp);
  auto seeded = std::make_shared<Var>(root_assign->var_->name_hint_ + "_seeded", assemble->GetType(), sp);
  auto scope = SpmdWrap(
      seed_index,
      {std::make_shared<AssignStmt>(fill_var, fill, sp), std::make_shared<AssignStmt>(seeded, assemble, sp)},
      MakeIndex(seed.work_units, sp), name + "_seed", sp);
  CubeSplitSeed result;
  result.scope = scope;
  result.output = seeded;
  return result;
}

bool EmitCubeMatmulInstance(const ::CubeSchedulePlan& plan, size_t instance,
                            const std::unordered_map<size_t, AssignStmtPtr>& op_assigns,
                            const ExprPtr& m_index, const ExprPtr& n_index, const ExprPtr& split_index,
                            const std::string& name, const Span& sp, std::vector<StmtPtr>* body,
                            std::vector<ExprPtr>* instance_values, std::vector<ExprPtr>* root_states,
                            std::string* reason) {
  const ::CubeMatmulSchedule& mm = plan.matmuls[instance];
  const AssignStmtPtr& source_assign = op_assigns.at(mm.op);
  auto source_call = As<Call>(source_assign->value_);
  auto output_type = As<TensorType>(source_assign->var_->GetType());
  auto fail = [&](std::string message) {
    *reason = std::move(message);
    return false;
  };
  if (source_call == nullptr || output_type == nullptr || source_call->args_.size() != 2) {
    return fail("planned matmul source is malformed");
  }
  const DataType storage_dtype = output_type->dtype_;
  auto lhs_source_type = AsTensorTypeLike(source_call->args_[0]->GetType());
  if (lhs_source_type == nullptr) return fail("matmul LHS is not tensor-like");
  const DataType accumulator_dtype = CubeAccumulatorDType(lhs_source_type->dtype_);

  ExprPtr lhs = source_call->args_[0];
  ExprPtr rhs = source_call->args_[1];
  if (mm.lhs_producer >= 0) {
    const size_t producer = static_cast<size_t>(mm.lhs_producer);
    if (!(*instance_values)[producer]) {
      return fail("missing planned LHS producer value");
    }
    auto original = AsVarLike(source_call->args_[0]);
    const AssignStmtPtr& producer_assign = op_assigns.at(plan.matmuls[producer].op);
    if (original == nullptr || original.get() != producer_assign->var_.get()) {
      return fail("planned LHS dependency does not match the source SSA edge");
    }
    lhs = (*instance_values)[producer];
  }
  if (mm.rhs_producer >= 0) {
    const size_t producer = static_cast<size_t>(mm.rhs_producer);
    if (!(*instance_values)[producer]) {
      return fail("missing planned RHS producer value");
    }
    auto original = AsVarLike(source_call->args_[1]);
    const AssignStmtPtr& producer_assign = op_assigns.at(plan.matmuls[producer].op);
    if (original == nullptr || original.get() != producer_assign->var_.get()) {
      return fail("planned RHS dependency does not match the source SSA edge");
    }
    rhs = (*instance_values)[producer];
  }

  ExprPtr lhs_row = mm.lhs_producer >= 0 ? MakeIndex(0, sp)
                                         : CubeRegionAxisOffset(mm.lhs.height_binding, mm.lhs.height, m_index,
                                                                n_index, split_index, sp);
  ExprPtr lhs_k = mm.lhs_producer >= 0 ? MakeIndex(0, sp)
                                       : CubeRegionAxisOffset(mm.lhs.width_binding, mm.lhs.width, m_index,
                                                              n_index, split_index, sp);
  ExprPtr rhs_k = mm.rhs_producer >= 0 ? MakeIndex(0, sp)
                                       : CubeRegionAxisOffset(mm.rhs.height_binding, mm.rhs.height, m_index,
                                                              n_index, split_index, sp);
  ExprPtr rhs_col = mm.rhs_producer >= 0 ? MakeIndex(0, sp)
                                         : CubeRegionAxisOffset(mm.rhs.width_binding, mm.rhs.width, m_index,
                                                                n_index, split_index, sp);
  const ExprPtr output_row =
      CubeRegionAxisOffset(mm.output.height_binding, mm.output.height, m_index, n_index, split_index, sp);
  const ExprPtr output_col =
      CubeRegionAxisOffset(mm.output.width_binding, mm.output.width, m_index, n_index, split_index, sp);

  auto find_variant = [&](int64_t h, int64_t w) -> const ::CubeOutputTileVariant* {
    for (const ::CubeOutputTileVariant& variant : mm.output_variants) {
      if (variant.height == h && variant.width == w) return &variant;
    }
    return nullptr;
  };

  // The legal PTO hierarchy is output/L0C tile outer, GM->L1 K-window inner.
  // A2/A3 cannot reload a Mat partial into Acc, so every child K loop keeps its
  // one [h,w] accumulator resident in L0C and drains exactly once afterwards.
  ExprPtr internal_state;
  if (!mm.is_sink) {
    auto scratch_call = OpRegistry::GetInstance().Create(
        "tensor.create_l1", {MakeIndexTuple({mm.output.height, mm.output.width}, sp)},
        {{"dtype", storage_dtype}, {"transpose", false}}, sp);
    auto scratch =
        std::make_shared<Var>(name + "_i" + std::to_string(instance) + "_l1", scratch_call->GetType(), sp);
    body->push_back(std::make_shared<AssignStmt>(scratch, scratch_call, sp));
    internal_state = scratch;
  }

  const int64_t tile_count = mm.output_tiles_m * mm.output_tiles_n;
  int64_t tile_index = 0;
  for (int64_t mj = 0; mj < mm.output_tiles_m; ++mj) {
    const int64_t mi = mj * mm.output_tile_m;
    const int64_t h = std::min(mm.output_tile_m, mm.output.height - mi);
    for (int64_t nj = 0; nj < mm.output_tiles_n; ++nj, ++tile_index) {
      const int64_t ni = nj * mm.output_tile_n;
      const int64_t w = std::min(mm.output_tile_n, mm.output.width - ni);
      const ::CubeOutputTileVariant* variant = find_variant(h, w);
      if (variant == nullptr) return fail("missing child L0 plan for an output boundary tile");

      const std::string tile_base =
          name + "_i" + std::to_string(instance) + "_t" + std::to_string(tile_index);
      auto tile_type = std::make_shared<TensorType>(std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)},
                                                    accumulator_dtype);
      auto tile_value = std::make_shared<Var>(tile_base + "_acc", tile_type, sp);
      for (StmtPtr& stmt : BuildTileMatmulAt(lhs, rhs, AddIndexOffset(lhs_row, MakeIndex(mi, sp), sp), lhs_k,
                                             rhs_k, AddIndexOffset(rhs_col, MakeIndex(ni, sp), sp), h, w,
                                             mm.effective_contraction, mm.k_loop.chunk, accumulator_dtype,
                                             tile_value, tile_base, sp, mm.k_loop.pipeline_stages,
                                             &variant->l0_init, &variant->l0_rolled, &variant->l0_tail)) {
        body->push_back(std::move(stmt));
      }

      if (mm.is_sink) {
        std::vector<std::pair<std::string, std::any>> kwargs;
        if (plan.split_k > 1) kwargs.emplace_back("atomic", 1);
        auto assemble = OpRegistry::GetInstance().Create(
            "tensor.assemble",
            {(*root_states)[instance], tile_value,
             MakeTuple2(AddIndexOffset(output_row, MakeIndex(mi, sp), sp),
                        AddIndexOffset(output_col, MakeIndex(ni, sp), sp), sp)},
            kwargs, sp);
        const bool last = tile_index + 1 == tile_count;
        auto next =
            last ? source_assign->var_ : std::make_shared<Var>(tile_base + "_gm", assemble->GetType(), sp);
        body->push_back(std::make_shared<AssignStmt>(next, assemble, sp));
        (*root_states)[instance] = next;
      } else {
        auto assemble = OpRegistry::GetInstance().Create(
            "tensor.assemble", {internal_state, tile_value, MakeIndexTuple({mi, ni}, sp)}, sp);
        auto next = std::make_shared<Var>(tile_base + "_l1", assemble->GetType(), sp);
        body->push_back(std::make_shared<AssignStmt>(next, assemble, sp));
        internal_state = next;
      }
    }
  }

  if (!mm.is_sink) (*instance_values)[instance] = internal_state;
  return true;
}

// Replay a solver-owned homogeneous cube request DAG for one exact, uniform
// SpatialSchedule candidate. The solver has already performed the recursive
// consumer-to-producer discovery; `plan.matmuls` is its memoized postorder, so
// emission is a linear producer-before-consumer replay. Every produced request
// gets a distinct local value, allowing one source matmul to be recomputed for
// different fan-out roles without conflating their regions.
std::optional<std::vector<StmtPtr>> EmitCubeScheduleGroup(
    const std::vector<StmtPtr>& members, const std::unordered_map<const Stmt*, size_t>& stmt_op,
    const SolverTile& tile, const std::string& name) {
  const ::CubeSchedulePlan& plan = tile.cube_schedule;
  auto decline = [&](const std::string& reason) {
    LOG_INFO << "AutoFuse[cube-plan]: group '" << name << "' decline — " << reason;
    return std::optional<std::vector<StmtPtr>>{};
  };
  std::string validation_error;
  auto validated = ValidateCubeScheduleGroup(members, stmt_op, tile, &validation_error);
  if (!validated) return decline(validation_error);
  const auto& op_assigns = validated->op_assigns;
  const int64_t parts_m = validated->parts_m;
  const int64_t parts_n = validated->parts_n;

  const Span sp = members.front()->span_;
  auto& registry = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
  std::vector<StmtPtr> result;
  std::vector<ExprPtr> root_initial(plan.matmuls.size());

  // Boundary outputs are the only cube values allocated outside the AIC body.
  // Internal request regions stay inside the AIC body. Their eventual L1
  // scratch and L0 subdivision are exclusively owned by AutoTileMatmulL0.
  for (size_t instance = 0; instance < plan.matmuls.size(); ++instance) {
    const ::CubeMatmulSchedule& mm = plan.matmuls[instance];
    if (!mm.is_sink) continue;
    const AssignStmtPtr& assign = op_assigns.at(mm.op);
    const auto [output_m, output_n] = Static2DShape(assign->var_->GetType());
    auto output_type = As<TensorType>(assign->var_->GetType());
    auto create = registry.Create("tensor.create", {MakeIndexTuple({output_m, output_n}, sp)},
                                  {{"dtype", output_type->dtype_}, {"layout", TensorLayout::ND}}, sp);
    auto initial = std::make_shared<Var>(assign->var_->name_hint_ + "_cube_out_" + std::to_string(instance),
                                         create->GetType(), sp);
    result.push_back(std::make_shared<AssignStmt>(initial, create, sp));
    root_initial[instance] = initial;
  }

  // Split-K's atomic target must be initialized before any AIC partial writes.
  if (plan.split_k > 1) {
    size_t root = 0;
    while (root < plan.matmuls.size() && !plan.matmuls[root].is_sink) ++root;
    CubeSplitSeed seed = BuildCubeSplitSeed(op_assigns.at(plan.matmuls[root].op), root_initial[root],
                                            plan.seed, parts_n, name, sp);
    result.push_back(std::move(seed.scope));
    root_initial[root] = std::move(seed.output);
  }

  auto work_index = std::make_shared<Var>(name + "_work", index_type, sp);
  ExprPtr split_index =
      plan.split_k > 1 ? MakeFloorMod(work_index, MakeIndex(plan.split_k, sp), sp) : MakeIndex(0, sp);
  ExprPtr spatial_index =
      plan.split_k > 1 ? MakeFloorDiv(work_index, MakeIndex(plan.split_k, sp), sp) : ExprPtr(work_index);
  ExprPtr m_index = MakeFloorDiv(spatial_index, MakeIndex(parts_n, sp), sp);
  ExprPtr n_index = MakeFloorMod(spatial_index, MakeIndex(parts_n, sp), sp);

  std::vector<StmtPtr> body;
  std::vector<ExprPtr> instance_values(plan.matmuls.size());
  std::vector<ExprPtr> root_states = root_initial;
  for (size_t instance = 0; instance < plan.matmuls.size(); ++instance) {
    std::string emit_error;
    if (!EmitCubeMatmulInstance(plan, instance, op_assigns, m_index, n_index, split_index, name, sp, &body,
                                &instance_values, &root_states, &emit_error)) {
      return decline(emit_error);
    }
  }

  result.push_back(SpmdWrap(work_index, std::move(body), MakeIndex(plan.work_units, sp), name, sp));
  LOG_INFO << "AutoFuse[cube-plan]: group '" << name << "' emitted " << plan.matmuls.size()
           << " request instance(s), grid=" << parts_m << "x" << parts_n << "x" << plan.split_k;
  return result;
}

// Replay the first buildable mixed schedule: one standard matmul followed by a
// linear elementwise epilogue.  The outer launch owns `active_groups` physical
// 910B clusters; every cluster runs a fixed inner loop over its successor
// spatial regions.  The function-level UP_DOWN split is intentionally attached
// here: LowerAutoVectorSplit turns the cube result into an AIV shard and halves
// only the vector sub-region, so the model's two-lane division is realized by
// the emitted IR rather than inferred from hardware topology.
std::optional<std::vector<StmtPtr>> EmitMixedScheduleGroup(
    const std::vector<StmtPtr>& members,
    const std::unordered_map<const Stmt*, size_t>& stmt_op,
    const std::unordered_set<const Var*>& required_live_outs,
    const SolverTile& tile, const std::string& name) {
  const ::MixedSchedulePlan& plan = tile.mixed_schedule;
  auto decline = [&](const std::string& reason) {
    LOG_INFO << "AutoFuse[mixed-plan]: group '" << name << "' decline — " << reason;
    return std::optional<std::vector<StmtPtr>>{};
  };
  if (!plan.feasible || !plan.emit_compatible || !plan.topology ||
      plan.mode != ::MixedPipelineMode::OneWay ||
      plan.topology->stages.size() != 2 || plan.topology->transfers.size() != 1 ||
      plan.topology->stages[0].engine != ::MixedEngine::Cube ||
      plan.topology->stages[1].engine != ::MixedEngine::Vector ||
      plan.topology->stages[0].ops.size() != 1) {
    return decline("not the buildable one-way C->V topology");
  }
  if (plan.split_k != 1 || plan.vector_split != ::MixedVectorSplit::Rows ||
      plan.vector_lanes != 2 || plan.fifos.size() != 1 ||
      plan.fifos[0].direction != ::MixedTransferDirection::CubeToVector ||
      plan.fifos[0].slot_count != 8 ||
      plan.fifos[0].reserved_bytes !=
          plan.fifos[0].slot_bytes * plan.fifos[0].slot_count ||
      plan.fifos[0].reserved_bytes > ReadHwParams().vec_capacity) {
    return decline("split/lane/FIFO descriptor is not the emitted C->V protocol");
  }
  if (plan.cube_window_k <= 0 || tile.k != plan.cube_window_k ||
      plan.vector_stage_kind != ::VectorStreamKind::Materialized ||
      plan.vector_stage_peak_ub_bytes <= 0 ||
      plan.vector_stage_peak_ub_bytes + plan.fifos[0].reserved_bytes >
          ReadHwParams().vec_capacity) {
    return decline("stage-local cube/vector plan is not materializable as described");
  }
  if (plan.model_overlap_granted != plan.overlap_implementable ||
      plan.loop.axis != ::MixedPipelineAxis::SpatialRegion ||
      plan.loop.items_per_spatial_tile != 1 || plan.loop.work_items != plan.spatial_tiles ||
      plan.loop.active_groups <= 0 || plan.loop.max_trips_per_group <= 0 ||
      plan.loop.min_trips_per_group != plan.loop.max_trips_per_group ||
      plan.loop.work_items != plan.loop.active_groups * plan.loop.max_trips_per_group) {
    return decline("logical work items do not form one fixed per-group loop");
  }
  if (plan.m_partition.num_big != 0 || plan.n_partition.num_big != 0 ||
      plan.m_partition.parts <= 0 || plan.n_partition.parts <= 0 ||
      plan.m_partition.big <= 0 || plan.n_partition.big <= 0 ||
      plan.m_partition.big % 2 != 0 ||
      plan.spatial_tiles != plan.m_partition.parts * plan.n_partition.parts) {
    return decline("mixed v0 requires one exact uniform even-row spatial grid");
  }

  std::unordered_map<size_t, AssignStmtPtr> by_op;
  for (const StmtPtr& stmt : members) {
    auto op = stmt_op.find(stmt.get());
    auto assign = As<AssignStmt>(stmt);
    if (op == stmt_op.end() || assign == nullptr || As<Call>(assign->value_) == nullptr) {
      return decline("group contains a non-call assignment");
    }
    by_op.emplace(op->second, assign);
  }

  const size_t matmul_op = plan.topology->stages[0].ops.front();
  auto matmul_it = by_op.find(matmul_op);
  if (matmul_it == by_op.end()) return decline("planned matmul has no source statement");
  const AssignStmtPtr& matmul_assign = matmul_it->second;
  auto matmul = As<Call>(matmul_assign->value_);
  if (matmul == nullptr || !IsOp(matmul, "tensor.matmul") || matmul->args_.size() != 2) {
    return decline("cube stage is not a standard binary tensor.matmul");
  }
  if (matmul->GetKwarg<bool>("a_trans", false) ||
      matmul->GetKwarg<bool>("b_trans", false) ||
      matmul->GetKwarg<bool>("c_matrix_nz", false)) {
    return decline("non-default matmul semantics are not in mixed v0");
  }
  const auto [M, K] = Static2DShape(matmul->args_[0]->GetType());
  const auto [rhs_k, N] = Static2DShape(matmul->args_[1]->GetType());
  const auto [out_m, out_n] = Static2DShape(matmul_assign->var_->GetType());
  auto lhs_type = AsTensorTypeLike(matmul->args_[0]->GetType());
  auto rhs_type = AsTensorTypeLike(matmul->args_[1]->GetType());
  auto matmul_type = As<TensorType>(matmul_assign->var_->GetType());
  if (M <= 0 || N <= 0 || K <= 0 || rhs_k != K || out_m != M || out_n != N ||
      M != plan.m_partition.parts * plan.m_partition.big ||
      N != plan.n_partition.parts * plan.n_partition.big) {
    return decline("matmul shape does not match the mixed spatial plan");
  }
  if (!lhs_type || !rhs_type || lhs_type->dtype_ != rhs_type->dtype_ ||
      !MixedCubeOperandDTypeSupported(lhs_type->dtype_) || !matmul_type ||
      !CubeOutputCanCarryKLoop(matmul->args_[0], matmul_type->dtype_)) {
    return decline("mixed v0 requires same supported operand dtype and accumulator result dtype");
  }

  std::vector<AssignStmtPtr> vector_ops;
  vector_ops.reserve(plan.topology->stages[1].ops.size());
  const Var* previous = matmul_assign->var_.get();
  for (size_t op : plan.topology->stages[1].ops) {
    auto it = by_op.find(op);
    if (it == by_op.end()) return decline("planned vector op has no source statement");
    auto call = As<Call>(it->second->value_);
    if (call == nullptr || DescribeVectorOp(call).capability != ::VectorOpCapability::Elementwise) {
      return decline("vector stage contains a non-elementwise operation");
    }
    const auto output_type = AsTensorTypeLike(it->second->var_->GetType());
    int previous_uses = 0;
    for (const ExprPtr& arg : call->args_) {
      auto var = AsVarLike(arg);
      if (var && var.get() == previous) ++previous_uses;
      const auto input_type = AsTensorTypeLike(arg->GetType());
      if (input_type && (!output_type || input_type->dtype_ != output_type->dtype_ ||
                         input_type->dtype_ != matmul_type->dtype_)) {
        return decline("vector tensor operands require an unmodeled dtype conversion");
      }
      const auto [arg_m, arg_n] = Static2DShape(arg->GetType());
      if (arg_m >= 0 && !((arg_m == 1 || arg_m == M) &&
                          (arg_n == 1 || arg_n == N))) {
        return decline("vector operand is not a scalar/full tile/broadcast tile");
      }
    }
    const auto [vec_m, vec_n] = Static2DShape(it->second->var_->GetType());
    if (previous_uses < 1 || vec_m != M || vec_n != N ||
        required_live_outs.count(previous) != 0) {
      return decline("vector epilogue is not a linear single-live-out chain");
    }
    vector_ops.push_back(it->second);
    previous = it->second->var_.get();
  }
  if (vector_ops.empty()) return decline("empty vector epilogue");
  const AssignStmtPtr& sink = vector_ops.back();

  const int64_t h = plan.m_partition.big;
  const int64_t w = plan.n_partition.big;
  auto sink_type = As<TensorType>(sink->var_->GetType());
  if (!matmul_type || !sink_type ||
      plan.fifos[0].valid_rows != h || plan.fifos[0].valid_cols != w ||
      plan.fifos[0].slot_bytes !=
          h * w * std::max<int64_t>(1, static_cast<int64_t>(matmul_type->dtype_.GetBit()) / 8)) {
    return decline("FIFO slot does not match the full cube result tile");
  }

  const Span sp = members.front()->span_;
  auto& registry = OpRegistry::GetInstance();
  auto index_type = std::make_shared<ScalarType>(DataType::INDEX);
  auto create = registry.Create("tensor.create", {MakeIndexTuple({M, N}, sp)},
                                {{"dtype", sink_type->dtype_}, {"layout", TensorLayout::ND}}, sp);
  auto initial = std::make_shared<Var>(sink->var_->name_hint_ + "_mixed_out", create->GetType(), sp);
  auto initial_assign = std::make_shared<AssignStmt>(initial, create, sp);

  auto block = std::make_shared<Var>(name + "_group", index_type, sp);
  auto trip = std::make_shared<Var>(name + "_trip", index_type, sp);
  const int64_t trips = plan.loop.max_trips_per_group;
  ExprPtr item = MakeAdd(MakeMul(block, MakeIndex(trips, sp), sp), trip, sp);
  ExprPtr mi = MakeMul(MakeFloorDiv(item, MakeIndex(plan.n_partition.parts, sp), sp),
                           MakeIndex(h, sp), sp);
  ExprPtr ni = MakeMul(MakeFloorMod(item, MakeIndex(plan.n_partition.parts, sp), sp),
                           MakeIndex(w, sp), sp);

  auto output_iter = std::make_shared<IterArg>(sink->var_->name_hint_ + "_mixed_state",
                                                initial->GetType(), initial, sp);
  auto cube_tile_type = std::make_shared<TensorType>(
      std::vector<ExprPtr>{MakeIndex(h, sp), MakeIndex(w, sp)}, matmul_type->dtype_);
  auto cube_tile = std::make_shared<Var>(matmul_assign->var_->name_hint_ + "_mixed_tile",
                                         cube_tile_type, sp);
  std::vector<StmtPtr> loop_body;
  const int64_t k_chunk = tile.k > 0 ? tile.k : K;
  const int pipeline_stages = K / std::max<int64_t>(1, k_chunk) >= 3 ? 2 : 1;
  for (StmtPtr& stmt : BuildTileMatmul(matmul->args_[0], matmul->args_[1], mi, ni, h, w, K,
                                       k_chunk, matmul_type->dtype_, cube_tile,
                                       matmul_assign->var_->name_hint_ + "_mixed", sp,
                                       pipeline_stages)) {
    loop_body.push_back(std::move(stmt));
  }

  std::unordered_map<const Var*, VarPtr> values;
  values.emplace(matmul_assign->var_.get(), cube_tile);
  std::unordered_map<const Var*, VarPtr> slices;
  VarPtr vector_tile;
  for (const AssignStmtPtr& source : vector_ops) {
    auto call = As<Call>(source->value_);
    std::vector<ExprPtr> args;
    args.reserve(call->args_.size());
    for (const ExprPtr& arg : call->args_) {
      auto var = AsVarLike(arg);
      if (var) {
        auto value = values.find(var.get());
        if (value != values.end()) {
          args.push_back(value->second);
          continue;
        }
      }
      const auto [arg_m, arg_n] = Static2DShape(arg->GetType());
      if (arg_m < 0) {
        args.push_back(arg);
        continue;
      }
      if (var) {
        auto cached = slices.find(var.get());
        if (cached != slices.end()) {
          args.push_back(cached->second);
          continue;
        }
      }
      const int64_t slice_h = arg_m == 1 ? 1 : h;
      const int64_t slice_w = arg_n == 1 ? 1 : w;
      ExprPtr row = arg_m == 1 ? MakeIndex(0, sp) : mi;
      ExprPtr col = arg_n == 1 ? MakeIndex(0, sp) : ni;
      auto slice = registry.Create("tensor.slice",
                                   {arg, MakeIndexTuple({slice_h, slice_w}, sp),
                                    MakeTuple2(row, col, sp)}, sp);
      auto sliced = std::make_shared<Var>(source->var_->name_hint_ + "_mixed_in",
                                          slice->GetType(), sp);
      loop_body.push_back(std::make_shared<AssignStmt>(sliced, slice, sp));
      if (var) slices.emplace(var.get(), sliced);
      args.push_back(sliced);
    }
    auto replay = registry.Create(call->op_->name_, args, call->kwargs_, sp);
    vector_tile = std::make_shared<Var>(source->var_->name_hint_ + "_mixed_tile",
                                        replay->GetType(), sp);
    loop_body.push_back(std::make_shared<AssignStmt>(vector_tile, replay, sp));
    values.emplace(source->var_.get(), vector_tile);
  }

  auto assemble = registry.Create("tensor.assemble",
                                  {ExprPtr(output_iter), vector_tile, MakeTuple2(mi, ni, sp)}, sp);
  auto next = std::make_shared<Var>(sink->var_->name_hint_ + "_mixed_next",
                                    assemble->GetType(), sp);
  loop_body.push_back(std::make_shared<AssignStmt>(next, assemble, sp));
  loop_body.push_back(std::make_shared<YieldStmt>(std::vector<ExprPtr>{next}, sp));

  auto item_loop = std::make_shared<ForStmt>(
      trip, MakeIndex(0, sp), MakeIndex(trips, sp), MakeIndex(1, sp),
      std::vector<IterArgPtr>{output_iter}, SeqStmts::Flatten(std::move(loop_body), sp),
      std::vector<VarPtr>{sink->var_}, sp,
      ForKind::Sequential, std::vector<std::pair<std::string, std::any>>{});

  std::vector<StmtPtr> kernel_body{
      std::make_shared<AssignStmt>(
          block, registry.Create("tile.get_block_idx", {}, sp), sp),
      item_loop};
  auto kernel = std::make_shared<InCoreScopeStmt>(
      std::optional<SplitMode>{SplitMode::UpDown}, name,
      SeqStmts::Flatten(std::move(kernel_body), sp), sp,
      std::vector<std::string>{},
      std::vector<std::pair<std::string, std::any>>{
          {"slot_num", static_cast<int>(plan.fifos[0].slot_count)}});
  INTERNAL_CHECK(kernel->HasAttr("slot_num")) << "mixed InCore lost slot_num at construction";
  auto scope = std::make_shared<SpmdScopeStmt>(
      MakeIndex(plan.loop.active_groups, sp), /*sync_start=*/false,
      name + "_spmd", kernel, sp);
  LOG_INFO << "AutoFuse[mixed-plan]: group '" << name << "' emitted C->V grid="
           << plan.m_partition.parts << "x" << plan.n_partition.parts
           << " groups=" << plan.loop.active_groups << " trips=" << trips
           << " lanes=" << plan.vector_lanes << " fifo_bytes=" << plan.fifos[0].slot_bytes
           << (plan.model_overlap_granted ? " fifo-overlap" : " serial");
  return std::vector<StmtPtr>{initial_assign, scope};
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
    } else if (auto sub = As<Submit>(a->value_)) {
      // Submit-aware (pass-submit-awareness rule): a Submit's uses are its args AND its deps_
      // (TaskId SSA values). A manual_scope body should not reach the flat auto_fuse DAG, but if
      // one does, the reorder's dependency edges must include deps_ or a TaskId use is dropped.
      for (const ExprPtr& arg : sub->args_) add(arg);
      for (const ExprPtr& dep : sub->deps_) add(dep);
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
StmtPtr EmitFusedScopes(const StmtPtr& body, const std::unordered_map<const Stmt*, size_t>& stmt_group,
                        const std::unordered_map<const Stmt*, SolverTile>& stmt_tile,
                        const std::unordered_map<const Stmt*, size_t>& stmt_op,
                        const std::unordered_map<const Stmt*, size_t>& stmt_exec,
                        const std::unordered_map<size_t, size_t>& group_p4_match,
                        const std::vector<P4Match>& p4_matches,
                        bool* mixed_emit_failed) {
  std::vector<StmtPtr> body_stmts;
  if (auto seq = As<SeqStmts>(body)) {
    body_stmts = seq->stmts_;
  } else {
    body_stmts.push_back(body);
  }

  std::unordered_set<const Var*> required_live_outs;
  for (const StmtPtr& stmt : body_stmts) {
    auto ret = As<ReturnStmt>(stmt);
    if (ret == nullptr) continue;
    for (const ExprPtr& value : ret->value_) {
      auto var = AsVarLike(value);
      if (var != nullptr) required_live_outs.insert(var.get());
    }
  }

  // Cluster each solver group into a contiguous run (dependency-preserving) so a group whose
  // members are interleaved with another group's in SSA order still emits as ONE fused scope.
  body_stmts = ReorderBodyByGroup(body_stmts, stmt_group);
  std::unordered_map<size_t, std::vector<StmtPtr>> group_members;
  for (const StmtPtr& stmt : body_stmts) {
    auto group = stmt_group.find(stmt.get());
    if (group != stmt_group.end()) group_members[group->second].push_back(stmt);
  }

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

  std::unordered_set<size_t> cube_fallback_groups;
  std::unordered_set<size_t> cube_plan_attempted;
  std::unordered_set<size_t> cube_plan_emitted;
  std::unordered_set<size_t> mixed_plan_attempted;
  std::unordered_set<size_t> mixed_plan_emitted;
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
        const P4Match* p4_match = nullptr;
        auto pit = group_p4_match.find(static_cast<size_t>(run_group));
        if (pit != group_p4_match.end() && pit->second < p4_matches.size()) {
          p4_match = &p4_matches[pit->second];
        }
        if (auto generic =
                EmitFusedGroupGeneric(run, tit->second, nm, p4_match, required_live_outs)) {
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
    auto tile_it = stmt_tile.find(stmt.get());
    if (GenericEmitEnabled() && MixedEmitEnabled() && tile_it != stmt_tile.end() &&
        tile_it->second.mixed_schedule.feasible) {
      const size_t mixed_group = static_cast<size_t>(g);
      if (mixed_plan_emitted.count(mixed_group) != 0) continue;
      if (mixed_plan_attempted.insert(mixed_group).second) {
        flush();
        auto members = group_members.find(mixed_group);
        if (members != group_members.end()) {
          if (auto emitted = EmitMixedScheduleGroup(
                  members->second, stmt_op, required_live_outs, tile_it->second,
                  "fused_" + std::to_string(g))) {
            for (StmtPtr& emitted_stmt : *emitted) top.push_back(std::move(emitted_stmt));
            mixed_plan_emitted.insert(mixed_group);
            continue;
          }
        }
        GenericDeclineB(
            "MixedSchedulePlan cannot be replayed exactly; leaving the function unchanged",
            stmt->span_);
        if (mixed_emit_failed != nullptr) *mixed_emit_failed = true;
        return body;
      }
    }
    if (GenericEmitEnabled() && tile_it != stmt_tile.end() && tile_it->second.cube_schedule.feasible &&
        !tile_it->second.cube_schedule.matmuls.empty()) {
      const size_t cube_group = static_cast<size_t>(g);
      if (cube_plan_emitted.count(cube_group) != 0) continue;
      if (cube_plan_attempted.insert(cube_group).second) {
        flush();
        auto members = group_members.find(cube_group);
        if (members != group_members.end()) {
          if (auto emitted = EmitCubeScheduleGroup(members->second, stmt_op, tile_it->second,
                                                   "fused_" + std::to_string(g))) {
            for (StmtPtr& emitted_stmt : *emitted) {
              top.push_back(std::move(emitted_stmt));
            }
            cube_plan_emitted.insert(cube_group);
            continue;
          }
        }
        cube_fallback_groups.insert(cube_group);
        GenericDeclineB(
            "CubeSchedulePlan cannot be replayed exactly; falling back to "
            "dependency-ordered standalone matmuls",
            stmt->span_);
      }
    }
    // A chained matmul pair (MM1 -> MM2 in the same group) is realized as one
    // fused kernel — the parallel-outer output tiling with the inner serial chain
    // and the intermediate on-chip (see TileChainedMatmul).
    auto hit = chain_head.find(stmt.get());
    if (hit != chain_head.end() && cube_fallback_groups.count(static_cast<size_t>(g)) == 0) {
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
    } else if (cube_fallback_groups.count(static_cast<size_t>(g)) != 0) {
      // A declined CubeSchedulePlan must become genuinely standalone kernels.
      // Keeping several fallback matmuls in one InCore scope would forward a
      // tile.matmul's FP32 L0C result directly to the next BF16/FP16 matmul,
      // bypassing the tensor-level store/narrow + reload boundary that makes
      // the fallback legal.  One scope per source op preserves its declared
      // tensor dtype and makes the materialization explicit.
      flush();
      const auto op_it = stmt_op.find(stmt.get());
      const std::string fallback_name = "fused_" + std::to_string(g) + "_fallback_" +
                                        std::to_string(op_it != stmt_op.end() ? op_it->second : 0);
      top.push_back(std::make_shared<InCoreScopeStmt>(std::nullopt, fallback_name, stmt, stmt->span_));
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
  // Existing user-written output params: a return that already ALIASES one of these is wired by
  // codegen (no action). But a return that is a NEW fused buffer, even alongside an existing Out
  // param, still needs lifting — a blanket "has any Out param -> skip" would leave that buffer
  // unwritten on the by-position harness (external-review finding). So collect the existing output
  // params and skip only the returns that reach them; lift the rest (appended AFTER these).
  std::unordered_set<const Var*> existing_out;
  for (size_t i = 0; i < func->params_.size() && i < func->param_directions_.size(); ++i)
    if (func->param_directions_[i] == ParamDirection::Out ||
        func->param_directions_[i] == ParamDirection::InOut)
      existing_out.insert(func->params_[i].get());
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

  // Walk the same return lineage; TRUE if it lands on an existing Out/InOut param (already wired by
  // codegen — leave it alone). Mirrors trace_to_create's chain (rebind / assemble arg0 /
  // set_validshape arg0 / tile.store arg2 / for-carry).
  auto reaches_out_param = [&](const VarPtr& ret_var) -> bool {
    if (existing_out.empty()) return false;
    const Var* cur = ret_var.get();
    std::unordered_set<const Var*> seen;
    while (cur != nullptr) {
      if (existing_out.count(cur) != 0) return true;
      if (!seen.insert(cur).second) break;
      if (auto it = idx.carry.find(cur); it != idx.carry.end()) { cur = it->second; continue; }
      auto dit = idx.var_def.find(cur);
      if (dit == idx.var_def.end()) break;
      const ExprPtr& val = dit->second->value_;
      if (auto rv = AsVarLike(val)) { cur = rv.get(); continue; }
      auto call = As<Call>(val);
      if (call == nullptr) break;
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
    return false;
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
  auto& reg = OpRegistry::GetInstance();
  const Span rsp = idx.ret->span_;
  std::vector<VarPtr> out_params;             // Out-param Var per return, in return order
  std::unordered_set<const Stmt*> drop_set;   // traceable creates to remove (their Var -> param)
  std::vector<StmtPtr> synth_copies;          // synthesized assemble-copies, inserted before return
  std::vector<ExprPtr> new_ret;               // rewritten return values
  for (const ExprPtr& rv : idx.ret->value_) {
    auto ret_var = AsVarLike(rv);
    if (!ret_var) return;
    if (reaches_out_param(ret_var)) {  // already an existing Out/InOut param -> codegen wires it
      new_ret.push_back(rv);
      continue;
    }
    auto ca = trace_to_create(ret_var);
    if (ca != nullptr) {
      // Traceable to the output buffer's `tensor.create`: move that Var into the param list (no
      // copy). Must be a distinct top-level child (safe to remove) — else bail (inconsistent ABI).
      if (top_level.count(ca.get()) == 0) return;
      if (!drop_set.insert(ca.get()).second) return;  // two returns share a buffer -> aliasing
      out_params.push_back(ca->var_);
      new_ret.push_back(rv);
    } else {
      // NOT traceable: the return is a computed value with no output `create`/`assemble` to move —
      // e.g. a returned `tensor.matmul` whose tiling DECLINED (non-uniform grid), left untiled. The
      // harness binds by param position, so append an Out param and COPY the computed result into it
      // with a full `tensor.assemble(out, val, [0,0])` (return_lineage traces the returned copy back
      // to the appended Out param). One extra full-tensor write; correctness over the fused-matmul
      // tiling gap. (Vector-ending fns hit the traceable branch above and pay no copy.)
      const auto [rM, rN] = Static2DShape(ret_var->GetType());
      if (rM < 0) return;  // dynamic / non-2D return -> out of scope
      (void)rN;
      auto out_var = std::make_shared<Var>(ret_var->name_hint_ + "_out", ret_var->GetType(), rsp);
      auto asm_call = reg.Create(
          "tensor.assemble",
          {ExprPtr(out_var), ExprPtr(ret_var), MakeTuple2(MakeIndex(0, rsp), MakeIndex(0, rsp), rsp)}, rsp);
      auto wr_var = std::make_shared<Var>(ret_var->name_hint_ + "_wr", asm_call->GetType(), rsp);
      synth_copies.push_back(std::make_shared<AssignStmt>(wr_var, asm_call, rsp));
      out_params.push_back(out_var);
      new_ret.push_back(ExprPtr(wr_var));
    }
  }

  // Every return already aliases an existing Out/InOut param -> nothing to wire (the old blanket
  // skip's common case, now reached only when it is actually safe).
  if (out_params.empty()) return;

  // Append the Out params (return order — the harness binds by position), drop the moved creates,
  // and insert the synthesized copies right before the (rewritten) return.
  for (const auto& v : out_params) {
    func->params_.push_back(v);
    func->param_directions_.push_back(ParamDirection::Out);
  }
  std::vector<StmtPtr> kept;
  kept.reserve(seq->stmts_.size() + synth_copies.size());
  for (const StmtPtr& s : seq->stmts_) {
    if (drop_set.count(s.get()) != 0) continue;  // dead create -> now the Out param
    if (As<ReturnStmt>(s)) {
      for (auto& w : synth_copies) kept.push_back(w);
      kept.push_back(std::make_shared<ReturnStmt>(new_ret, s->span_));
    } else {
      kept.push_back(s);
    }
  }
  func->body_ = SeqStmts::Flatten(std::move(kept), seq->span_);

  LOG_INFO << "AutoFuse[" << func->name_ << "]: wired " << out_params.size()
           << " return(s) -> appended Out param(s) (device/harness binds outputs by param)";
}

// Merge-decision knob (cost-vs-latency experiment, Round 3): override the solver's Phase-1
// PARTITION — which ops fuse into one group — with a fixed one, so the device can measure
// alternative fusion boundaries against the model's prediction (the partition-layer analog of
// PYPTO_AUTOFUSE_FORCE_PLAN). Env PYPTO_AUTOFUSE_FORCE_MERGE:
//   unset / "solver" -> the solver's argmin partition (::solve, the default; no override).
//   "none"           -> Partition::trivial (each op its own group = fully UNFUSED baseline).
//   "all"            -> one group holding every op (fully FUSED). If the ops cannot unify under a
//                       single grid, finalize() prices the group 1e18 and the emit declines — that
//                       infeasibility IS the answer "this cannot be fully fused", recorded honestly.
// The chosen partition is finalized and lowered via Solution::from_partition (the SAME ordering +
// costing the solver's Phase 2 uses), so its total_latency is the model's cost for that merge.
::Solution SolveWithMergeOverride(const ::Problem& prob, const ::DAG& dag, const std::string& fn) {
  const char* merge = std::getenv("PYPTO_AUTOFUSE_FORCE_MERGE");
  if (merge == nullptr || std::string(merge) == "solver") return ::solve(prob, dag);
  const std::string mode = merge;
  if (mode != "none" && mode != "all") {
    LOG_WARN << "AutoFuse[" << fn << "]: unknown PYPTO_AUTOFUSE_FORCE_MERGE='" << mode
             << "' (want none|all|solver) -> using the solver argmin partition";
    return ::solve(prob, dag);
  }
  ::Partition part = ::Partition::trivial(prob, dag);  // singletons (fully unfused)
  if (mode == "all") {
    std::vector<size_t> all_ops(prob.num_ops());
    for (size_t i = 0; i < prob.num_ops(); ++i) all_ops[i] = i;
    part.groups.assign(1, ::Partition::Group{});
    part.groups[0].ops = FlatSet<size_t>(all_ops.begin(), all_ops.end());
    part.groups[0].alive = true;
  }
  part.finalize();  // rebuild_index + per-group Subgraph::create/best_cost + rebuild_group_dag
  ::Solution sol = ::Solution::from_partition(prob, dag, part);
  LOG_INFO << "AutoFuse[" << fn << "]: FORCED MERGE '" << mode << "' -> " << sol.num_steps()
           << " group(s), total latency " << sol.total_latency();
  return sol;
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
    // Normalize inline-returned compute exprs into named bindings so the solver graph and the emit
    // both see EVERY op (a bare `return pl.op(...)` is otherwise invisible to ProblemBuilder — its
    // operands look group-internal and get dropped, BUG-LN-2). No-op for already-named returns.
    FunctionPtr wfunc = func;
    if (auto hoisted = HoistInlineReturnComputeExprs(func->body_)) {
      auto mut = MutableCopy(func);
      mut->body_ = *hoisted;
      wfunc = mut;
    }
    ProblemBuilder builder;
    builder.Build(wfunc, prog);
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
             << " Acc=" << p.cube_capacity << " UB=" << p.vec_capacity << " cube_l0_cost="
             << (p.use_hierarchical_cube_cost ? "exact" : "analytic");
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

    // Opaque/unsupported operations are partition barriers, but the current
    // solver requires every source op to belong to a costed group. A singleton
    // opaque group has no vector/cube iteration frame; letting it enter tile
    // enumeration therefore creates a fictional 0x0 schedule. Until AutoFuse
    // can solve supported DAG segments around an opaque barrier, decline the
    // whole function and leave every source operation to the existing lowering.
    const bool has_unsupported_ops =
        std::any_of(p.ops.begin(), p.ops.end(), [](const ::Op& op) {
          return op.type == ::OpType::Opaque &&
                 op.vector_capability == ::VectorOpCapability::Unsupported;
        });
    if (has_unsupported_ops) {
      LOG_INFO << "AutoFuse[" << func->name_
               << "]: unsupported opaque operation present -> leaving function unchanged";
      if (const char* dump_dir = std::getenv("PYPTO_AUTOFUSE_DUMP")) {
        DumpProblemJson(builder.problem,
                        std::string(dump_dir) + "/" + func->name_ + ".dag.json");
      }
      new_functions.emplace(gvar, func);
      continue;
    }

    // Solve, then print the fusion decision: each group's member ops + chosen tile.
    ::DAG dag = ::DAG::build(builder.problem);
    ::Solution sol = SolveWithMergeOverride(builder.problem, dag, func->name_);
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

    // A mixed group may only appear when the dedicated gate admitted the exact
    // C->V compiler subset.  Keep this as a Tier-B backstop against a solver
    // policy regression while the gate is off.
    if (GenericEmitEnabled() && !MixedEmitEnabled()) {
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
                              " appeared while PYPTO_AUTOFUSE_MIXED is disabled",
                          builder.op_stmts[gops.front()]->span_);
      }
    }

    // Emit: map each solver op back to its source statement, wrap each fused
    // group in an InCoreScopeStmt, and realize the chosen tile (step.config) for
    // matmuls — the output `[w,h]` tiling + the per-tile k-pipeline.
    // Cost-vs-wall-time validation knobs (off the hot path — only when set):
    //   PYPTO_AUTOFUSE_DUMP_PLANS=1       -> log every feasible candidate + modeled cost per group,
    //       as the FULL plan key: w,h,split,parts_m,parts_n (the solver's spatial grid, not just the
    //       tile). Two candidates can share (w,h,split) but differ in (parts_m,parts_n) -> different
    //       modeled cost, same emitted kernel; dumping the grid makes that collapse visible.
    //   PYPTO_AUTOFUSE_FORCE_PLAN="[g<N>:]w,h,s[,pm,pn]" -> EMIT that plan instead of the solver
    //       argmin, so the device runs it; the paired modeled cost is the logged FORCED line. A field
    //       = -1 (e.g. "-1,6,2") wildcards it. parts_m/parts_n are optional (default wildcard) —
    //       supply them to disambiguate two candidates sharing (w,h,split). An optional "g<N>:" prefix
    //       restricts the force to group N (else every matching group is forced — vary ONE group in a
    //       multi-group kernel with "g1:16,32,2"). The plan must be a feasible candidate.
    static const bool dump_plans = std::getenv("PYPTO_AUTOFUSE_DUMP_PLANS") != nullptr;
    static const char* force_env = std::getenv("PYPTO_AUTOFUSE_FORCE_PLAN");
    long fw = -1, fh = -1, fs = -1, fpm = -1, fpn = -1, fg = -1;
    if (force_env != nullptr) {
      const char* spec = force_env;
      if (spec[0] == 'g') std::sscanf(spec, "g%ld:", &fg);        // optional group selector
      if (const char* colon = std::strchr(spec, ':')) spec = colon + 1;
      std::sscanf(spec, "%ld,%ld,%ld,%ld,%ld", &fw, &fh, &fs, &fpm, &fpn);
    }

    std::unordered_map<const Stmt*, size_t> stmt_group;
    std::unordered_map<const Stmt*, SolverTile> stmt_tile;  // group's [w,h,k] tile, for matmul tiling
    std::unordered_map<const Stmt*, size_t> stmt_op;        // solver Problem op id
    std::unordered_map<const Stmt*, size_t> stmt_exec;      // solver's per-group pebbling order
    std::unordered_map<size_t, size_t> group_p4_match;      // solver group -> shared semantic descriptor
    std::map<FlatSet<size_t>, size_t> p4_match_by_ops;
    for (size_t i = 0; i < builder.problem.p4_patterns.size(); ++i) {
      p4_match_by_ops.emplace(builder.problem.p4_patterns[i].ops, i);
    }
    for (size_t s = 0; s < sol.num_steps(); ++s) {
      const ::TileConfig& cfg = sol.step(s).config;
      const ::CostResult& selected_cost = sol.step_cost(s);
      // CostResult is the hot local-search cache value and deliberately carries
      // no emit descriptor. Reconstruct the stream algorithm only for this final
      // winning configuration, using the same helper candidate pricing used.
      const ::VectorStreamPlan selected_stream =
          !sol.step(s).subgraph.has_matmul()
              ? sol.step(s).subgraph.vector_stream_plan(
                    cfg, sol.retained_entering(s), sol.step(s).retain_these)
              : ::VectorStreamPlan{};
      const ::CubeSchedulePlan selected_cube =
          sol.step(s).subgraph.has_matmul() && !sol.step(s).subgraph.has_vector()
              ? sol.step(s).subgraph.cube_schedule_plan(
                    cfg, sol.retained_entering(s), sol.step(s).retain_these,
                    selected_cost.parallel_split)
              : ::CubeSchedulePlan{};
      const ::MixedSchedulePlan selected_mixed =
          sol.step(s).subgraph.is_mixed()
              ? sol.step(s).subgraph.mixed_schedule_plan(
                    cfg, sol.retained_entering(s), sol.step(s).retain_these,
                    selected_cost.parallel_split)
              : ::MixedSchedulePlan{};
      SolverTile tile(cfg.w, cfg.h,
                      selected_cube.feasible
                          ? selected_cube.config.k
                          : (selected_mixed.feasible ? selected_mixed.config.k : cfg.k),
                      selected_cost.parallel_split, cfg.parts_m, cfg.parts_n,
                      selected_stream, selected_cube, selected_mixed);
      if (dump_plans || force_env != nullptr) {
        bool forced_here = false;
        std::set<std::tuple<int64_t, int64_t, size_t, int64_t, int64_t>> seen_plans;  // dedup identical keys
        for (const auto& [pc, pr] : sol.step(s).subgraph.enumerate_plans()) {
          if (dump_plans &&
              seen_plans.emplace(pc.w, pc.h, pr.parallel_split, pc.parts_m, pc.parts_n).second)
            LOG_INFO << "AutoFuse[" << func->name_ << "]: PLAN group=" << s << " w=" << pc.w << " h="
                     << pc.h << " split=" << pr.parallel_split << " parts_m=" << pc.parts_m
                     << " parts_n=" << pc.parts_n << " cost=" << pr.latency;
          if (force_env != nullptr && !forced_here && (fg < 0 || static_cast<long>(s) == fg) &&
              (fw < 0 || pc.w == fw) && (fh < 0 || pc.h == fh) &&
              (fs < 0 || static_cast<long>(pr.parallel_split) == fs) &&
              (fpm < 0 || pc.parts_m == fpm) && (fpn < 0 || pc.parts_n == fpn)) {
            const ::VectorStreamPlan forced_stream =
                !sol.step(s).subgraph.has_matmul()
                    ? sol.step(s).subgraph.vector_stream_plan(pc)
                    : ::VectorStreamPlan{};
            const ::CubeSchedulePlan forced_cube =
                sol.step(s).subgraph.has_matmul() && !sol.step(s).subgraph.has_vector()
                    ? sol.step(s).subgraph.cube_schedule_plan(
                          pc, {}, {}, pr.parallel_split)
                    : ::CubeSchedulePlan{};
            const ::MixedSchedulePlan forced_mixed =
                sol.step(s).subgraph.is_mixed()
                    ? sol.step(s).subgraph.mixed_schedule_plan(
                          pc, {}, {}, pr.parallel_split)
                    : ::MixedSchedulePlan{};
            tile = SolverTile(pc.w, pc.h,
                              forced_cube.feasible
                                  ? forced_cube.config.k
                                  : (forced_mixed.feasible ? forced_mixed.config.k : pr.config.k),
                              pr.parallel_split, pc.parts_m, pc.parts_n,
                              forced_stream, forced_cube, forced_mixed);
            forced_here = true;
            LOG_INFO << "AutoFuse[" << func->name_ << "]: FORCED group=" << s << " w=" << pc.w << " h="
                     << pc.h << " split=" << pr.parallel_split << " parts_m=" << pc.parts_m
                     << " parts_n=" << pc.parts_n << " cost=" << pr.latency;
          }
        }
        if (force_env != nullptr && !forced_here && (fg < 0 || static_cast<long>(s) == fg))
          LOG_WARN << "AutoFuse[" << func->name_ << "]: FORCE_PLAN '" << force_env
                   << "' matched NO feasible candidate for group=" << s
                   << " -> using the solver argmin (experiment plan is NOT the one you forced)";
      }
      for (size_t op_idx : sol.step(s).subgraph.ops()) {
        const Stmt* stmt = builder.op_stmts[op_idx];
        stmt_group[stmt] = s;
        stmt_tile[stmt] = tile;
        stmt_op[stmt] = op_idx;
      }
      const FlatSet<size_t> group_ops(sol.step(s).subgraph.ops().begin(), sol.step(s).subgraph.ops().end());
      auto p4_it = p4_match_by_ops.find(group_ops);
      if (p4_it != p4_match_by_ops.end()) {
        INTERNAL_CHECK(p4_it->second < builder.p4_matches.size())
            << "Internal error: P4 solver pattern has no matching IR descriptor";
        group_p4_match.emplace(s, p4_it->second);
      }
      // The solver's execution_order() is the depth-first pebbling order it costed the
      // working-set peak along — the order the emit MUST replay to stay within UB.
      const std::vector<size_t>& exec = sol.step(s).subgraph.execution_order();
      for (size_t pos = 0; pos < exec.size(); ++pos) {
        stmt_exec[builder.op_stmts[exec[pos]]] = pos;
      }
    }
    auto new_func = MutableCopy(wfunc);
    bool mixed_emit_failed = false;
    new_func->body_ = EmitFusedScopes(wfunc->body_, stmt_group, stmt_tile, stmt_op, stmt_exec,
                                      group_p4_match, builder.p4_matches,
                                      &mixed_emit_failed);
    if (mixed_emit_failed) {
      new_functions.emplace(gvar, func);
      continue;
    }
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
