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

#include <cstdint>
#include <limits>

#include "pypto/backend/common/backend_handler.h"
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

/** Solver-owned algorithm descriptor for the first standalone cube surface. */
struct CubeSchedulePlan {
  bool feasible = false;
  CubeSpatialPolicy spatial_policy = CubeSpatialPolicy::Uniform;
  int64_t parts_m = 0;
  int64_t parts_n = 0;
  int64_t region_m = 0;
  int64_t region_n = 0;
  int64_t work_units = 0;
  int64_t peak_l1_bytes = 0;
  int64_t gm_to_l1_bytes_per_work_unit = 0;
  int64_t gm_to_l1_bytes_total = 0;
  double modeled_gm_to_l1_cycles = 0.0;
  double modeled_l0_cycles = 0.0;
  double modeled_cycles = std::numeric_limits<double>::infinity();
  utils::L0TileResult l0_plan;
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
