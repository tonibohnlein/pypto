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

#include <cstdint>
#include <exception>
#include <map>
#include <optional>
#include <string>
#include <unordered_set>
#include <utility>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/core/dtype.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/memory_space.h"
#include "pypto/ir/pipe.h"
#include "pypto/ir/program.h"
#include "pypto/ir/span.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/mutable_copy.h"
#include "src/ir/transforms/auto_tile/cube_emit.h"
#include "src/ir/transforms/auto_tile/cube_graph.h"
#include "src/ir/transforms/auto_tile/cube_plan.h"
#include "src/ir/transforms/auto_tile/vector_emit.h"
#include "src/ir/transforms/auto_tile/vector_graph.h"
#include "src/ir/transforms/auto_tile/vector_plan.h"
#include "src/ir/transforms/auto_tile/vector_report.h"

namespace pypto {
namespace ir {
namespace pass {
namespace {

class CalledFunctionCollector : public IRVisitor {
 public:
  explicit CalledFunctionCollector(const Program* program) : program_(program) {}

  [[nodiscard]] const std::unordered_set<std::string>& called() const { return called_; }

 protected:
  void VisitExpr_(const CallPtr& call) override {
    if (call != nullptr && call->op_ != nullptr && program_->GetFunction(call->op_->name_) != nullptr) {
      called_.insert(call->op_->name_);
    }
    IRVisitor::VisitExpr_(call);
  }

  void VisitExpr_(const SubmitPtr& submit) override {
    if (submit != nullptr && submit->op_ != nullptr && program_->GetFunction(submit->op_->name_) != nullptr) {
      called_.insert(submit->op_->name_);
    }
    IRVisitor::VisitExpr_(submit);
  }

 private:
  const Program* program_;
  std::unordered_set<std::string> called_;
};

struct VectorTargetContext {
  auto_tile::VectorHardware hardware;
  const backend::BackendHandler* handler = nullptr;
};

struct CubeTargetContext {
  auto_tile::CubeHardware hardware;
};

const backend::BackendHandler* ReadBackendHandler(const Span& span) {
  CHECK_SPAN(backend::BackendConfig::IsConfigured(), span)
      << "AutoTile requires an explicitly configured backend";
  const backend::Backend* backend = backend::GetBackend();
  const PassContext* context = PassContext::Current();
  const backend::BackendHandler* handler =
      context != nullptr ? context->GetBackendHandler() : backend->GetHandler();
  CHECK_SPAN(handler != nullptr, span) << "AutoTile could not obtain the active backend handler";
  return handler;
}

VectorTargetContext ReadVectorTarget(const Span& span) {
  const backend::BackendHandler* handler = ReadBackendHandler(span);
  const backend::Backend* backend = backend::GetBackend();
  const std::optional<backend::VectorAutoTile910BTarget> target = handler->GetVectorAutoTile910BTarget();
  CHECK_SPAN(target.has_value(), span) << "AutoTile vector scheduling currently supports Ascend910B only";
  VectorTargetContext result;
  result.handler = handler;
  result.hardware.vector_cores = backend->GetCoreCount(CoreType::VECTOR);
  result.hardware.ub_bytes = static_cast<int64_t>(backend->GetMemSize(MemorySpace::Vec));
  result.hardware.dma_alignment_bytes = target->dma_alignment_bytes;
  result.hardware.vector_register_bytes = target->vector_register_bytes;
  CHECK_SPAN(result.hardware.vector_cores > 0 && result.hardware.ub_bytes > 0 &&
                 result.hardware.dma_alignment_bytes > 0 && result.hardware.vector_register_bytes > 0,
             span)
      << "AutoTile could not derive the " << target->model_name << " vector topology";
  return result;
}

CubeTargetContext ReadCubeTarget(const Span& span, const DataType& accumulator_dtype) {
  const backend::BackendHandler* handler = ReadBackendHandler(span);
  const backend::Backend* backend = backend::GetBackend();
  const std::optional<backend::CubeAutoTile910BTarget> target = handler->GetCubeAutoTile910BTarget();
  CHECK_SPAN(target.has_value(), span) << "AutoTile cube scheduling currently supports Ascend910B only";
  CubeTargetContext result;
  result.hardware.cube_cores = backend->GetCoreCount(CoreType::CUBE);
  result.hardware.l1_bytes = static_cast<int64_t>(handler->GetMatCapacityBytes());
  result.hardware.fractal = handler->GetL0FractalAlignment();
  result.hardware.min_l0_tile = handler->GetMinL0TileDim();
  result.hardware.l0c_m_alignment = handler->GetL0cMAlignment(accumulator_dtype);
  result.hardware.l0a_bytes = handler->GetL0aCapacityBytes();
  result.hardware.l0b_bytes = handler->GetL0bCapacityBytes();
  result.hardware.l0c_bytes = handler->GetL0cCapacityBytes();
  result.hardware.core_frequency_hz = target->core_frequency_hz;
  result.hardware.gm_to_l1_bandwidth_gib_per_s = target->gm_to_l1_bandwidth_gib_per_s;
  const PassContext* context = PassContext::Current();
  result.hardware.allow_double_buffer_c =
      context != nullptr &&
      (context->GetMemoryPlanner() == MemoryPlanner::PtoAS || context->GetEnablePyptoL0cDoubleBuffer());
  result.hardware.l0_cost = handler->GetL0CostModel();
  CHECK_SPAN(result.hardware.cube_cores > 0 && result.hardware.l1_bytes > 0 && result.hardware.fractal > 0 &&
                 result.hardware.min_l0_tile > 0 && result.hardware.l0a_bytes > 0 &&
                 result.hardware.l0b_bytes > 0 && result.hardware.l0c_bytes > 0 &&
                 result.hardware.core_frequency_hz > 0.0 &&
                 result.hardware.gm_to_l1_bandwidth_gib_per_s > 0.0,
             span)
      << "AutoTile could not derive the " << target->model_name << " cube topology";
  return result;
}

ProgramPtr TransformAutoTile(const ProgramPtr& program) {
  if (program == nullptr) return program;
  CalledFunctionCollector calls(program.get());
  for (const auto& [unused, function] : program->functions_) {
    (void)unused;
    if (function != nullptr && function->body_ != nullptr) calls.VisitStmt(function->body_);
  }

  std::map<GlobalVarPtr, FunctionPtr, GlobalVarPtrLess> functions;
  bool changed = false;
  for (const auto& [global, function] : program->functions_) {
    if (function == nullptr || !function->GetAttr<bool>("auto_tile", false)) {
      functions.emplace(global, function);
      continue;
    }
    CHECK_SPAN(function->func_type_ == FunctionType::Opaque, function->span_)
        << "AutoTile must run on a tensor-level Opaque function before scope outlining";
    FunctionPtr emitted;
    if (auto_tile::ContainsCubeOperation(function)) {
      auto_tile::CubeAdmissionResult admission = auto_tile::AdmitCubeGraph(function, program);
      if (!admission.supported && admission.failure != nullptr) std::rethrow_exception(admission.failure);
      CHECK_SPAN(admission.supported, function->span_) << "AutoTile cannot admit the entire marked function";
      const auto_tile::CubeGraph& graph = admission.graph;
      const CubeTargetContext target = ReadCubeTarget(function->span_, graph.accumulator_dtype);
      const auto_tile::CubeSchedulePlan plan = auto_tile::CubePlanner910B(target.hardware).Plan(graph);
      CHECK_SPAN(plan.feasible, function->span_)
          << "AutoTile cannot realize the entire marked function as one capacity-safe Ascend910B cube kernel";
      emitted = auto_tile::EmitCubeSchedule(graph, plan, calls.called());
      int64_t retained_panel_count = static_cast<int64_t>(plan.resident_boundaries.size());
      int64_t retained_l1_bytes = 0;
      for (const auto_tile::CubeResidentBoundaryPlan& resident : plan.resident_boundaries) {
        retained_l1_bytes += resident.bytes;
      }
      if (plan.matmuls.empty()) {
        retained_panel_count += static_cast<int64_t>(plan.retain_lhs) + static_cast<int64_t>(plan.retain_rhs);
        retained_l1_bytes += plan.retained_lhs_bytes + plan.retained_rhs_bytes;
      } else {
        for (const auto_tile::CubeMatmulSchedule& request : plan.matmuls) {
          retained_panel_count +=
              static_cast<int64_t>(request.retain_lhs) + static_cast<int64_t>(request.retain_rhs);
          retained_l1_bytes += request.retained_lhs_bytes + request.retained_rhs_bytes;
        }
      }
      LOG_INFO << "AutoTile[" << function->name_ << "]: cube schedule="
               << (plan.serial_dag()
                       ? "serial_dag"
                       : (plan.k_loop.pipeline_stages >= 2 ? "k_window_pipeline" : "serial_matmul"))
               << " grid=" << plan.parts_m << "x" << plan.parts_n << "x" << plan.split_k
               << " requests=" << (plan.matmuls.empty() ? 1 : plan.matmuls.size())
               << " work_units=" << plan.work_units << " spatial_work_units=" << plan.spatial_work_units
               << " region=" << plan.region_m << "x" << plan.region_n << " output_tile=" << plan.output_tile_m
               << "x" << plan.output_tile_n << " output_tiles=" << plan.output_tiles_m << "x"
               << plan.output_tiles_n
               << " spatial_policy=" << auto_tile::CubeSpatialPolicyName(plan.spatial_policy)
               << " split_merge=" << (plan.first_partial_then_atomic() ? "first_partial_then_atomic" : "none")
               << " split_phases=" << plan.first_partial_work_units << "+" << plan.atomic_rest_work_units
               << " retained_panels=" << retained_panel_count << " retained_l1=" << retained_l1_bytes
               << " peak_l1=" << plan.peak_l1_bytes << " gm_to_l1_bytes=" << plan.gm_to_l1_bytes_total
               << " k_window=" << plan.k_loop.l1_window_k << " chunk=" << plan.k_loop.chunk
               << " chunks=" << plan.k_loop.full_chunks << "+" << plan.k_loop.tail
               << " stages=" << plan.k_loop.pipeline_stages << " l0_tile=" << plan.l0_init.m << "x"
               << plan.l0_init.n << "x" << plan.l0_init.k
               << " gm_to_l1_cycles=" << plan.modeled_gm_to_l1_cycles
               << " l0_cycles=" << plan.modeled_l0_cycles
               << " final_drain_cycles=" << plan.modeled_final_drain_cycles
               << " split_sync_cycles=" << plan.modeled_split_sync_cycles
               << " modeled_cycles=" << plan.modeled_cycles;
    } else {
      const VectorTargetContext target = ReadVectorTarget(function->span_);
      auto_tile::VectorAdmissionResult admission =
          auto_tile::AdmitVectorGraph(function, program, target.handler->GetTcvtAdjacency());
      if (!admission.supported && admission.failure != nullptr) std::rethrow_exception(admission.failure);
      CHECK_SPAN(admission.supported, function->span_) << "AutoTile cannot admit the entire marked function";
      const auto_tile::VectorGraph& graph = admission.graph;
      const auto_tile::VectorSchedulePlan plan = auto_tile::VectorPlanner910B(target.hardware).Plan(graph);
      CHECK_SPAN(plan.feasible, function->span_) << "AutoTile cannot realize the entire marked function as "
                                                    "one capacity-safe Ascend910B vector kernel";
      emitted = auto_tile::EmitVectorSchedule(graph, plan, calls.called());
      const std::optional<std::string> report_path = auto_tile::WriteVectorScheduleReport(graph, plan);
      LOG_INFO << "AutoTile[" << function->name_
               << "]: vector schedule=" << auto_tile::ScheduleKindName(plan.kind)
               << " grid=" << plan.m_partition.parts << "x" << plan.n_partition.parts
               << " work_units=" << plan.work_units << " tile=" << plan.tile_h << "x" << plan.tile_w
               << " strip=" << plan.strip_h << "x" << plan.strip_w << " chunks=" << plan.full_chunks << "x"
               << plan.chunk << "+" << plan.tail << " largest_feasible_chunk=" << plan.largest_feasible_chunk
               << " feasible_chunks=" << plan.feasible_chunk_candidates << " stages="
               << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Body)].pipeline_stages << "/"
               << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Stats)].pipeline_stages << "/"
               << plan.phases[auto_tile::PhaseIndex(auto_tile::VectorPhase::Apply)].pipeline_stages
               << " peak_ub=" << plan.chunk_peak_ub_bytes << " full_peak_ub=" << plan.full_peak_ub_bytes
               << " compute_cycles=" << plan.modeled_compute_cycles
               << " transfer_cycles=" << plan.modeled_transfer_cycles << " phase_compute=["
               << plan.modeled_phase_compute_cycles[0] << "," << plan.modeled_phase_compute_cycles[1] << ","
               << plan.modeled_phase_compute_cycles[2] << "," << plan.modeled_phase_compute_cycles[3] << "]"
               << " phase_transfer=[" << plan.modeled_phase_transfer_cycles[0] << ","
               << plan.modeled_phase_transfer_cycles[1] << "," << plan.modeled_phase_transfer_cycles[2] << ","
               << plan.modeled_phase_transfer_cycles[3] << "]"
               << " phase_input_bytes=[" << static_cast<int64_t>(plan.modeled_phase_input_bytes[0]) << ","
               << static_cast<int64_t>(plan.modeled_phase_input_bytes[1]) << ","
               << static_cast<int64_t>(plan.modeled_phase_input_bytes[2]) << ","
               << static_cast<int64_t>(plan.modeled_phase_input_bytes[3]) << "]"
               << " phase_output_bytes=[" << static_cast<int64_t>(plan.modeled_phase_output_bytes[0]) << ","
               << static_cast<int64_t>(plan.modeled_phase_output_bytes[1]) << ","
               << static_cast<int64_t>(plan.modeled_phase_output_bytes[2]) << ","
               << static_cast<int64_t>(plan.modeled_phase_output_bytes[3]) << "]"
               << " reduction_model=" << (plan.used_reduction_fallback ? "legacy_fallback" : "grounded")
               << " pointwise_model=" << auto_tile::PointwiseCostModelName(plan)
               << " modeled_cycles=" << plan.modeled_cycles
               << (report_path.has_value() ? " report=" + *report_path : "");
    }
    functions.emplace(global, emitted);
    changed = true;
  }
  if (!changed) return program;
  auto result = MutableCopy(program);
  result->functions_ = std::move(functions);
  return result;
}

}  // namespace

Pass AutoTile() { return CreateProgramPass(TransformAutoTile, "AutoTile", kAutoTileProperties); }

}  // namespace pass
}  // namespace ir
}  // namespace pypto
