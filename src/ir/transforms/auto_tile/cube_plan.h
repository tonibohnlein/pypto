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

#ifndef SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_PLAN_H_
#define SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_PLAN_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#include "pypto/backend/common/backend_handler.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "src/ir/transforms/auto_tile/cube_graph.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {

enum class CubeSpatialPolicy : uint8_t {
  Uniform,
  ClampedOverlap,
};

enum class CubeAxisBinding : uint8_t {
  Full,
  SpatialM,
  SpatialN,
};

enum class CubeOperandRole : uint8_t {
  Lhs,
  Rhs,
};

/** One role-expanded tensor region in the serial request DAG. */
struct CubeTensorRegionPlan {
  VarPtr tensor;
  int64_t height = 0;
  int64_t width = 0;
  CubeAxisBinding height_binding = CubeAxisBinding::Full;
  CubeAxisBinding width_binding = CubeAxisBinding::Full;
};

/** Solver-owned outer GM->L1 contraction stream for one matmul request. */
struct CubeKLoopPlan {
  int64_t l1_window_k = 0;
  int64_t chunk = 0;
  int64_t full_chunks = 0;
  int64_t tail = 0;
  int pipeline_stages = 1;

  [[nodiscard]] bool streamed() const { return full_chunks >= 2; }
};

/** One compatible boundary region retained from first through last request. */
struct CubeResidentBoundaryPlan {
  size_t id = std::numeric_limits<size_t>::max();
  CubeTensorRegionPlan region;
  CubeOperandRole role = CubeOperandRole::Lhs;
  size_t first_use = 0;
  size_t last_use = 0;
  size_t use_count = 0;
  int64_t bytes = 0;
};

/** One serial matmul request in producer-before-consumer execution order. */
struct CubeMatmulSchedule {
  size_t instance = std::numeric_limits<size_t>::max();
  size_t node = std::numeric_limits<size_t>::max();
  CubeOperandRole consumer_role = CubeOperandRole::Lhs;
  int64_t lhs_producer = -1;
  int64_t rhs_producer = -1;
  int64_t lhs_resident_boundary = -1;
  int64_t rhs_resident_boundary = -1;
  CubeTensorRegionPlan lhs;
  CubeTensorRegionPlan rhs;
  CubeTensorRegionPlan output;
  bool is_sink = false;
  bool retain_lhs = false;
  bool retain_rhs = false;
  int64_t retained_lhs_bytes = 0;
  int64_t retained_rhs_bytes = 0;
  int64_t output_tile_m = 0;
  int64_t output_tile_n = 0;
  int64_t output_tiles_m = 0;
  int64_t output_tiles_n = 0;
  int64_t peak_transient_l1_bytes = 0;
  int64_t gm_to_l1_bytes = 0;
  double modeled_cycles = 0.0;
  double modeled_l0_cycles = 0.0;
  double modeled_drain_cycles = 0.0;
  CubeKLoopPlan k_loop;
  utils::L0TileResult l0_init;
  utils::L0TileResult l0_rolled;
  utils::L0TileResult l0_tail;
};

/** Solver-owned algorithm descriptor for the first standalone cube surface. */
struct CubeSchedulePlan {
  bool feasible = false;
  CubeSpatialPolicy spatial_policy = CubeSpatialPolicy::Uniform;
  int64_t parts_m = 0;
  int64_t parts_n = 0;
  int64_t region_m = 0;
  int64_t region_n = 0;
  int64_t output_tile_m = 0;
  int64_t output_tile_n = 0;
  int64_t output_tiles_m = 0;
  int64_t output_tiles_n = 0;
  int64_t work_units = 0;
  int64_t split_k = 1;
  int64_t spatial_work_units = 0;
  int64_t first_partial_work_units = 0;
  int64_t atomic_rest_work_units = 0;
  int64_t peak_l1_bytes = 0;
  int64_t gm_to_l1_bytes_per_work_unit = 0;
  int64_t gm_to_l1_bytes_total = 0;
  bool retain_lhs = false;
  bool retain_rhs = false;
  int64_t retained_lhs_bytes = 0;
  int64_t retained_rhs_bytes = 0;
  CubeKLoopPlan k_loop;
  double modeled_gm_to_l1_cycles = 0.0;
  double modeled_l0_cycles = 0.0;
  double modeled_final_drain_cycles = 0.0;
  double modeled_split_sync_cycles = 0.0;
  double modeled_cycles = std::numeric_limits<double>::infinity();
  utils::L0TileResult l0_init;
  utils::L0TileResult l0_rolled;
  utils::L0TileResult l0_tail;
  std::vector<CubeMatmulSchedule> matmuls;
  std::vector<CubeResidentBoundaryPlan> resident_boundaries;
  std::vector<size_t> execution_order;

  [[nodiscard]] bool first_partial_then_atomic() const { return split_k > 1; }
  [[nodiscard]] bool serial_dag() const { return matmuls.size() > 1; }
};

struct CubeHardware {
  int cube_cores = 0;
  int64_t l1_bytes = 0;
  int fractal = 0;
  int min_l0_tile = 0;
  int l0c_m_alignment = 0;
  uint32_t l0a_bytes = 0;
  uint32_t l0b_bytes = 0;
  uint32_t l0c_bytes = 0;
  double core_frequency_hz = 0.0;
  double gm_to_l1_bandwidth_gib_per_s = 0.0;
  bool allow_double_buffer_c = false;
  backend::L0CostModel l0_cost;
};

class CubePlanner910B {
 public:
  explicit CubePlanner910B(CubeHardware hardware) : hardware_(hardware) {}

  [[nodiscard]] CubeSchedulePlan Plan(const CubeGraph& graph) const;

 private:
  CubeHardware hardware_;
};

[[nodiscard]] const char* CubeSpatialPolicyName(CubeSpatialPolicy policy);

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto

#endif  // SRC_IR_TRANSFORMS_AUTO_TILE_CUBE_PLAN_H_
