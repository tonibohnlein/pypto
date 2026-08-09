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

#include "src/ir/transforms/auto_tile/cube_plan.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/error.h"

namespace pypto {
namespace ir {
namespace pass {
namespace auto_tile {
namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

int64_t CeilDiv(int64_t value, int64_t divisor) { return (value + divisor - 1) / divisor; }

int64_t AlignUp(int64_t value, int64_t alignment) { return CeilDiv(value, alignment) * alignment; }

int64_t DTypeBytes(const DataType& dtype) {
  return std::max<int64_t>(1, static_cast<int64_t>(dtype.GetByte()));
}

std::vector<std::pair<int64_t, int64_t>> CandidateRegions(int64_t extent, int64_t alignment,
                                                          int64_t max_parts) {
  std::set<std::pair<int64_t, int64_t>> unique;
  for (int64_t requested_parts = 1; requested_parts <= max_parts; ++requested_parts) {
    const int64_t region = AlignUp(CeilDiv(extent, requested_parts), alignment);
    // An oversized one-region clamp is not buildable: there is no valid source
    // slice of that static shape. Multiple clamped regions may overlap, but each
    // region itself must fit inside the logical extent.
    if (region < alignment || region > extent) continue;
    unique.emplace(region, CeilDiv(extent, region));
  }
  return {unique.begin(), unique.end()};
}

utils::L0TileConfig MakeL0Config(const CubeGraph& graph, const CubeHardware& hardware, int64_t region_m,
                                 int64_t region_n) {
  utils::L0TileConfig config;
  config.M = static_cast<int>(region_m);
  config.N = static_cast<int>(region_n);
  config.K = static_cast<int>(graph.k);
  config.l0a_bytes = hardware.l0a_bytes;
  config.l0b_bytes = hardware.l0b_bytes;
  config.l0c_bytes = hardware.l0c_bytes;
  config.bytes_a = static_cast<uint32_t>(DTypeBytes(graph.operand_dtype));
  config.bytes_b = static_cast<uint32_t>(DTypeBytes(graph.operand_dtype));
  config.bytes_c = static_cast<uint32_t>(DTypeBytes(graph.accumulator_dtype));
  config.min_m = hardware.min_l0_tile;
  config.min_n = hardware.min_l0_tile;
  config.min_k = hardware.min_l0_tile;
  config.align_m = hardware.fractal;
  config.align_n = hardware.fractal;
  config.align_k = hardware.fractal;
  config.l0c_align_m = hardware.l0c_m_alignment;
  config.allow_a_stationary = true;
  config.allow_b_stationary = true;
  config.allow_double_buffer_c = hardware.allow_double_buffer_c;
  config.allow_padding = false;
  config.allow_k_boundary = true;
  config.bw_a = hardware.l0_cost.bw_l0a;
  config.bw_b = hardware.l0_cost.bw_l0b;
  config.bw_drain = hardware.l0_cost.bw_drain;
  config.drain_fixed_cycles = hardware.l0_cost.drain_fixed_cycles;
  config.drain_row_cycles = hardware.l0_cost.drain_row_cycles;
  config.drain_penalty_cycles = hardware.l0_cost.drain_penalty_cycles;
  config.drain_c0_bytes = hardware.l0_cost.drain_c0_bytes;
  config.mad_head = hardware.l0_cost.mad_head_cycles;
  config.mad_k_fractal_bytes = hardware.l0_cost.mad_k_fractal_bytes;
  config.mad_fp32_passes = hardware.l0_cost.mad_fp32_passes;
  return config;
}

}  // namespace

CubeSchedulePlan CubePlanner910B::Plan(const CubeGraph& graph) const {
  CubeSchedulePlan best;
  if (hardware_.cube_cores <= 0 || hardware_.l1_bytes <= 0 || hardware_.fractal <= 0 ||
      hardware_.core_frequency_hz <= 0.0 || hardware_.gm_to_l1_bandwidth_gib_per_s <= 0.0 ||
      graph.m < hardware_.fractal || graph.n < hardware_.fractal || graph.k < hardware_.fractal ||
      graph.k % hardware_.fractal != 0) {
    return best;
  }

  const int64_t max_work_units = 2 * static_cast<int64_t>(hardware_.cube_cores);
  const auto m_regions = CandidateRegions(graph.m, hardware_.fractal, max_work_units);
  const auto n_regions = CandidateRegions(graph.n, hardware_.fractal, max_work_units);
  const int64_t operand_bytes = DTypeBytes(graph.operand_dtype);
  const double gm_cycles_per_byte =
      hardware_.core_frequency_hz / (kGiB * hardware_.gm_to_l1_bandwidth_gib_per_s);

  auto better = [](const CubeSchedulePlan& lhs, const CubeSchedulePlan& rhs) {
    if (!rhs.feasible) return true;
    return std::tie(lhs.modeled_cycles, lhs.work_units, lhs.region_m, lhs.region_n) <
           std::tie(rhs.modeled_cycles, rhs.work_units, rhs.region_m, rhs.region_n);
  };

  for (const auto& [region_m, parts_m] : m_regions) {
    for (const auto& [region_n, parts_n] : n_regions) {
      const int64_t work_units = parts_m * parts_n;
      if (work_units <= 0 || work_units > max_work_units) continue;
      const int64_t l1_bytes = (region_m * graph.k + graph.k * region_n) * operand_bytes;
      if (l1_bytes <= 0 || l1_bytes > hardware_.l1_bytes) continue;

      utils::L0TileResult l0;
      try {
        l0 = utils::ChooseL0Tile(MakeL0Config(graph, hardware_, region_m, region_n));
      } catch (const pypto::ValueError&) {
        continue;
      }
      if (l0.m <= 0 || l0.n <= 0 || l0.k <= 0) continue;

      CubeSchedulePlan candidate;
      candidate.feasible = true;
      candidate.parts_m = parts_m;
      candidate.parts_n = parts_n;
      candidate.region_m = region_m;
      candidate.region_n = region_n;
      candidate.work_units = work_units;
      candidate.spatial_policy = parts_m * region_m == graph.m && parts_n * region_n == graph.n
                                     ? CubeSpatialPolicy::Uniform
                                     : CubeSpatialPolicy::ClampedOverlap;
      candidate.peak_l1_bytes = l1_bytes;
      candidate.gm_to_l1_bytes_per_work_unit = l1_bytes;
      candidate.gm_to_l1_bytes_total = l1_bytes * work_units;
      candidate.modeled_gm_to_l1_cycles = static_cast<double>(l1_bytes) * gm_cycles_per_byte;
      candidate.modeled_l0_cycles = static_cast<double>(l0.estimated_cost_cycles);
      const int64_t waves = CeilDiv(work_units, hardware_.cube_cores);
      // This first surface emits one serial GM->L1 operand phase followed by
      // the existing L0 matmul algorithm. There is intentionally no overlap
      // term until the outer K-window pipeline is part of CubeSchedulePlan.
      candidate.modeled_cycles =
          static_cast<double>(waves) * (candidate.modeled_gm_to_l1_cycles + candidate.modeled_l0_cycles);
      candidate.l0_plan = l0;
      if (std::isfinite(candidate.modeled_cycles) && better(candidate, best)) best = candidate;
    }
  }
  return best;
}

const char* CubeSpatialPolicyName(CubeSpatialPolicy policy) {
  switch (policy) {
    case CubeSpatialPolicy::Uniform:
      return "uniform";
    case CubeSpatialPolicy::ClampedOverlap:
      return "clamped_overlap";
  }
  return "unknown";
}

}  // namespace auto_tile
}  // namespace pass
}  // namespace ir
}  // namespace pypto
