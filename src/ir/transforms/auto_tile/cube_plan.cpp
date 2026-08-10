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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/core/dtype.h"
#include "pypto/core/error.h"
#include "pypto/core/logging.h"
#include "pypto/ir/expr.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/l0_tile_chooser.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/auto_tile/cube_graph.h"

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
                                 int64_t region_n, int64_t contraction, bool allow_operand_stationary,
                                 bool c_read) {
  utils::L0TileConfig config;
  config.M = static_cast<int>(region_m);
  config.N = static_cast<int>(region_n);
  config.K = static_cast<int>(contraction);
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
  config.allow_a_stationary = allow_operand_stationary;
  config.allow_b_stationary = allow_operand_stationary;
  config.allow_double_buffer_c = hardware.allow_double_buffer_c;
  config.c_read = c_read;
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

int64_t PipelinedChunk(int64_t extent, int64_t l1_window, int64_t fractal,
                       int64_t physical_stream_buffers = 2) {
  if (extent < 3 * fractal || l1_window < 2 * fractal) return 0;
  if (physical_stream_buffers <= 0) return 0;
  const int64_t limit = std::min(l1_window / physical_stream_buffers, extent / 3);
  const int64_t aligned = (limit / fractal) * fractal;
  return aligned >= fractal ? aligned : 0;
}

double KWindowStreamWall(int64_t full_chunks, int pipeline_stages, double first_feed, double first_work,
                         double rolled_feed, double rolled_work) {
  if (full_chunks <= 0) return 0.0;
  const int64_t rolled = full_chunks - 1;
  if (rolled == 0) return first_feed + first_work;
  if (pipeline_stages < 2) {
    return first_feed + first_work + static_cast<double>(rolled) * (rolled_feed + rolled_work);
  }
  return first_feed + std::max(first_work, rolled_feed) +
         static_cast<double>(rolled - 1) * std::max(rolled_work, rolled_feed) + rolled_work;
}

double L0WorkCycles(const CubeGraph& graph, const CubeHardware& hardware, const utils::L0TileResult& l0,
                    int64_t region_m, int64_t region_n, int64_t contraction) {
  const int64_t tiles_m = CeilDiv(region_m, l0.m);
  const int64_t tiles_n = CeilDiv(region_n, l0.n);
  const int64_t operand_bytes = DTypeBytes(graph.operand_dtype);
  const int64_t k_fractal = std::max<int64_t>(1, hardware.l0_cost.mad_k_fractal_bytes / operand_bytes);
  const double compute_passes =
      graph.operand_dtype == DataType::FP32 ? static_cast<double>(hardware.l0_cost.mad_fp32_passes) : 1.0;
  const double mad_per_tile = static_cast<double>(hardware.l0_cost.mad_head_cycles) +
                              compute_passes * static_cast<double>(CeilDiv(l0.m, hardware.fractal)) *
                                  static_cast<double>(CeilDiv(contraction, k_fractal)) *
                                  static_cast<double>(CeilDiv(l0.n, hardware.fractal));
  const double mad = static_cast<double>(tiles_m * tiles_n) * mad_per_tile;
  const double lhs_bytes = static_cast<double>(region_m * contraction * tiles_n * operand_bytes);
  const double rhs_bytes = static_cast<double>(contraction * region_n * tiles_m * operand_bytes);
  const double extracts = lhs_bytes / hardware.l0_cost.bw_l0a + rhs_bytes / hardware.l0_cost.bw_l0b;
  return std::max(mad, extracts);
}

double FinalDrainCycles(const CubeGraph& graph, const CubeHardware& hardware, const utils::L0TileResult& l0,
                        int64_t region_m, int64_t region_n) {
  const int64_t tiles_m = CeilDiv(region_m, l0.m);
  const int64_t tiles_n = CeilDiv(region_n, l0.n);
  const int64_t bytes_c = DTypeBytes(graph.accumulator_dtype);
  const int64_t n0 = std::max<int64_t>(1, hardware.l0_cost.drain_c0_bytes / bytes_c);
  const int64_t n1 = CeilDiv(l0.n, n0);
  const int64_t extra_passes = std::max<int64_t>(0, (n1 + 1) / 2 - 1);
  const double per_row = std::max(hardware.l0_cost.drain_row_cycles,
                                  static_cast<double>(bytes_c * l0.n) / hardware.l0_cost.bw_drain) +
                         hardware.l0_cost.drain_penalty_cycles * static_cast<double>(extra_passes);
  const double per_tile = hardware.l0_cost.drain_fixed_cycles + static_cast<double>(l0.m) * per_row;
  return static_cast<double>(tiles_m * tiles_n) * per_tile;
}

CubeGraph RequestGraphView(const CubeGraph& graph, const CubeMatmulNode& node, int64_t output_m,
                           int64_t output_n) {
  CubeGraph view = graph;
  view.m = output_m;
  view.n = output_n;
  view.k = node.k;
  view.operand_dtype = node.operand_dtype;
  view.accumulator_dtype = node.accumulator_dtype;
  view.storage_dtype = node.storage_dtype;
  return view;
}

struct RequestKey {
  size_t node = 0;
  CubeOperandRole consumer_role = CubeOperandRole::Lhs;
  int64_t height = 0;
  int64_t width = 0;
  CubeAxisBinding height_binding = CubeAxisBinding::Full;
  CubeAxisBinding width_binding = CubeAxisBinding::Full;

  [[nodiscard]] auto tie() const {
    return std::tie(node, consumer_role, height, width, height_binding, width_binding);
  }
  [[nodiscard]] bool operator<(const RequestKey& other) const { return tie() < other.tie(); }
};

std::vector<CubeMatmulSchedule> BuildRequestDag(const CubeGraph& graph, int64_t root_m, int64_t root_n) {
  std::vector<CubeMatmulSchedule> requests;
  std::map<RequestKey, size_t> memo;
  std::function<size_t(size_t, CubeOperandRole, const CubeTensorRegionPlan&)> visit =
      [&](size_t node_index, CubeOperandRole consumer_role, const CubeTensorRegionPlan& output) -> size_t {
    const RequestKey key{node_index,   consumer_role,         output.height,
                         output.width, output.height_binding, output.width_binding};
    if (auto found = memo.find(key); found != memo.end()) return found->second;

    const CubeMatmulNode& node = graph.matmuls[node_index];
    CubeTensorRegionPlan lhs{node.lhs, output.height, node.k, output.height_binding, CubeAxisBinding::Full};
    CubeTensorRegionPlan rhs{node.rhs, node.k, output.width, CubeAxisBinding::Full, output.width_binding};
    int64_t lhs_producer = -1;
    int64_t rhs_producer = -1;
    if (node.lhs_producer >= 0) {
      lhs_producer =
          static_cast<int64_t>(visit(static_cast<size_t>(node.lhs_producer), CubeOperandRole::Lhs, lhs));
    }
    if (node.rhs_producer >= 0) {
      rhs_producer =
          static_cast<int64_t>(visit(static_cast<size_t>(node.rhs_producer), CubeOperandRole::Rhs, rhs));
    }

    CubeMatmulSchedule request;
    request.instance = requests.size();
    request.node = node_index;
    request.consumer_role = consumer_role;
    request.lhs_producer = lhs_producer;
    request.rhs_producer = rhs_producer;
    request.lhs = std::move(lhs);
    request.rhs = std::move(rhs);
    request.output = output;
    request.is_sink = node_index == graph.sink && output.height_binding == CubeAxisBinding::SpatialM &&
                      output.width_binding == CubeAxisBinding::SpatialN;
    const size_t instance = requests.size();
    requests.push_back(std::move(request));
    memo.emplace(key, instance);
    return instance;
  };

  const CubeTensorRegionPlan root{graph.output, root_m, root_n, CubeAxisBinding::SpatialM,
                                  CubeAxisBinding::SpatialN};
  // The sink drains to GM and therefore has no consumer-side Mat role. LHS is
  // a stable sentinel; recursively requested internal values use this field as
  // part of their physical request identity.
  visit(graph.sink, CubeOperandRole::Lhs, root);
  return requests;
}

struct BoundaryKey {
  const Var* tensor = nullptr;
  CubeOperandRole role = CubeOperandRole::Lhs;
  int64_t height = 0;
  int64_t width = 0;
  CubeAxisBinding height_binding = CubeAxisBinding::Full;
  CubeAxisBinding width_binding = CubeAxisBinding::Full;

  [[nodiscard]] auto tie() const {
    return std::tie(tensor, role, height, width, height_binding, width_binding);
  }
  [[nodiscard]] bool operator<(const BoundaryKey& other) const { return tie() < other.tie(); }
};

CubeAxisBinding CanonicalBoundaryBinding(CubeAxisBinding binding, int64_t requested_extent,
                                         const ExprPtr& tensor, size_t axis) {
  auto type = tensor != nullptr ? As<TensorType>(tensor->GetType()) : nullptr;
  if (type == nullptr || axis >= type->shape_.size()) return binding;
  auto extent = As<ConstInt>(type->shape_[axis]);
  if (extent != nullptr && extent->value_ == requested_extent) {
    return CubeAxisBinding::Full;
  }
  return binding;
}

std::vector<CubeResidentBoundaryPlan> DeriveResidentBoundaries(std::vector<CubeMatmulSchedule>* requests) {
  struct UseSummary {
    CubeTensorRegionPlan region;
    CubeOperandRole role = CubeOperandRole::Lhs;
    std::vector<std::pair<size_t, bool>> uses;
  };
  std::map<BoundaryKey, UseSummary> summaries;
  for (size_t index = 0; index < requests->size(); ++index) {
    CubeMatmulSchedule& request = (*requests)[index];
    auto record = [&](const CubeTensorRegionPlan& region, CubeOperandRole role, int64_t producer, bool lhs) {
      if (producer >= 0) return;
      BoundaryKey key{region.tensor.get(),
                      role,
                      region.height,
                      region.width,
                      CanonicalBoundaryBinding(region.height_binding, region.height, region.tensor, 0),
                      CanonicalBoundaryBinding(region.width_binding, region.width, region.tensor, 1)};
      auto& summary = summaries[key];
      summary.region = region;
      summary.role = role;
      summary.uses.emplace_back(index, lhs);
    };
    record(request.lhs, CubeOperandRole::Lhs, request.lhs_producer, true);
    record(request.rhs, CubeOperandRole::Rhs, request.rhs_producer, false);
  }

  std::vector<CubeResidentBoundaryPlan> residents;
  for (auto& [key, summary] : summaries) {
    (void)key;
    if (summary.uses.size() < 2) continue;
    CubeResidentBoundaryPlan resident;
    resident.id = residents.size();
    resident.region = summary.region;
    resident.role = summary.role;
    resident.first_use = summary.uses.front().first;
    resident.last_use = summary.uses.back().first;
    resident.use_count = summary.uses.size();
    auto tensor = As<TensorType>(summary.region.tensor->GetType());
    INTERNAL_CHECK(tensor != nullptr) << "Internal error: cube resident boundary is not a Tensor";
    resident.bytes = summary.region.height * summary.region.width * DTypeBytes(tensor->dtype_);
    for (const auto& [request_index, lhs] : summary.uses) {
      int64_t& slot = lhs ? (*requests)[request_index].lhs_resident_boundary
                          : (*requests)[request_index].rhs_resident_boundary;
      slot = static_cast<int64_t>(resident.id);
    }
    residents.push_back(std::move(resident));
  }
  return residents;
}

std::optional<CubeMatmulSchedule> PlanFixedRequest(const CubeGraph& graph,
                                                   const CubeMatmulSchedule& descriptor,
                                                   const CubeHardware& hardware,
                                                   int64_t persistent_l1_bytes) {
  const CubeMatmulNode& node = graph.matmuls[descriptor.node];
  const CubeGraph view = RequestGraphView(graph, node, descriptor.output.height, descriptor.output.width);
  const int64_t operand_bytes = DTypeBytes(node.operand_dtype);
  const int64_t accumulator_bytes = DTypeBytes(node.accumulator_dtype);
  const double gm_cycles_per_byte =
      hardware.core_frequency_hz / (kGiB * hardware.gm_to_l1_bandwidth_gib_per_s);
  std::optional<CubeMatmulSchedule> best;

  for (int retained_mask = 0; retained_mask < 4; ++retained_mask) {
    CubeMatmulSchedule candidate = descriptor;
    const bool lhs_boundary = candidate.lhs_producer < 0 && candidate.lhs_resident_boundary < 0;
    const bool rhs_boundary = candidate.rhs_producer < 0 && candidate.rhs_resident_boundary < 0;
    candidate.retain_lhs = lhs_boundary && (retained_mask & 1) != 0;
    candidate.retain_rhs = rhs_boundary && (retained_mask & 2) != 0;
    if ((retained_mask & 1) != 0 && !lhs_boundary) continue;
    if ((retained_mask & 2) != 0 && !rhs_boundary) continue;
    candidate.retained_lhs_bytes =
        candidate.retain_lhs ? candidate.lhs.height * candidate.lhs.width * operand_bytes : 0;
    candidate.retained_rhs_bytes =
        candidate.retain_rhs ? candidate.rhs.height * candidate.rhs.width * operand_bytes : 0;

    const bool lhs_cached = !lhs_boundary || candidate.retain_lhs;
    const bool rhs_cached = !rhs_boundary || candidate.retain_rhs;
    const int64_t retained_bytes = candidate.retained_lhs_bytes + candidate.retained_rhs_bytes;
    if (persistent_l1_bytes + retained_bytes > hardware.l1_bytes) continue;
    const int64_t streamed_elements =
        (lhs_cached ? 0 : candidate.lhs.height) + (rhs_cached ? 0 : candidate.rhs.width);
    const int64_t available = hardware.l1_bytes - persistent_l1_bytes - retained_bytes;
    const int64_t l1_window =
        streamed_elements == 0
            ? node.k
            : std::min<int64_t>(node.k, (available / (streamed_elements * operand_bytes) / hardware.fractal) *
                                            hardware.fractal);
    if (l1_window < hardware.fractal) continue;

    candidate.k_loop.l1_window_k = l1_window;
    // A serial-DAG request emits a prologue load for the first K window and a
    // two-slot rolled pipeline. Lifetime reuse may fold the prologue into one
    // rolled slot, but that is an optional allocator optimization rather than
    // an emitter contract. Size the chunk for all three physical buffers so a
    // feasible plan remains feasible under the actual lowered allocation.
    const int64_t pipelined_chunk =
        streamed_elements > 0 ? PipelinedChunk(node.k, l1_window, hardware.fractal, 3) : 0;
    if (pipelined_chunk > 0) {
      candidate.k_loop.chunk = pipelined_chunk;
      candidate.k_loop.pipeline_stages = 2;
    } else {
      candidate.k_loop.chunk =
          streamed_elements == 0 ? node.k : std::max<int64_t>(hardware.fractal, std::min(l1_window, node.k));
      candidate.k_loop.pipeline_stages = 1;
    }
    candidate.k_loop.full_chunks = node.k / candidate.k_loop.chunk;
    candidate.k_loop.tail = node.k - candidate.k_loop.full_chunks * candidate.k_loop.chunk;
    if (candidate.k_loop.full_chunks <= 0 || candidate.k_loop.tail % hardware.fractal != 0) {
      continue;
    }

    const bool one_full_contraction = candidate.k_loop.full_chunks == 1 && candidate.k_loop.tail == 0;
    try {
      candidate.l0_init =
          utils::ChooseL0Tile(MakeL0Config(view, hardware, descriptor.output.height, descriptor.output.width,
                                           one_full_contraction ? node.k : candidate.k_loop.chunk,
                                           /*allow_operand_stationary=*/false, /*c_read=*/false));
      candidate.l0_rolled =
          one_full_contraction
              ? candidate.l0_init
              : utils::ChooseL0Tile(MakeL0Config(view, hardware, descriptor.output.height,
                                                 descriptor.output.width, candidate.k_loop.chunk,
                                                 /*allow_operand_stationary=*/false, /*c_read=*/true));
      if (candidate.k_loop.tail > 0) {
        candidate.l0_tail = utils::ChooseL0Tile(MakeL0Config(
            view, hardware, descriptor.output.height, descriptor.output.width, candidate.k_loop.tail,
            /*allow_operand_stationary=*/false, /*c_read=*/true));
      }
    } catch (const pypto::ValueError&) {
      continue;
    }
    candidate.output_tile_m = std::min<int64_t>(candidate.l0_init.m, candidate.l0_rolled.m);
    candidate.output_tile_n = std::min<int64_t>(candidate.l0_init.n, candidate.l0_rolled.n);
    if (candidate.k_loop.tail > 0) {
      candidate.output_tile_m = std::min<int64_t>(candidate.output_tile_m, candidate.l0_tail.m);
      candidate.output_tile_n = std::min<int64_t>(candidate.output_tile_n, candidate.l0_tail.n);
    }
    // The first partial and the carried accumulated result are distinct SSA
    // identities until semantic accumulator reuse coalesces them.  Keep the
    // tensor-level plan buildable before that downstream lifetime rewrite by
    // reserving at most half of L0C for either identity.
    const int64_t half_l0c_elements = static_cast<int64_t>(hardware.l0c_bytes) / (2 * accumulator_bytes);
    while (candidate.output_tile_m * candidate.output_tile_n > half_l0c_elements) {
      if (candidate.output_tile_m >= candidate.output_tile_n && candidate.output_tile_m > hardware.fractal) {
        candidate.output_tile_m -= hardware.fractal;
      } else if (candidate.output_tile_n > hardware.fractal) {
        candidate.output_tile_n -= hardware.fractal;
      } else {
        candidate.output_tile_m = 0;
        candidate.output_tile_n = 0;
        break;
      }
    }
    if (candidate.output_tile_m <= 0 || candidate.output_tile_n <= 0) continue;

    // AutoTileMatmulL0 independently replays the emitted child request.  The
    // first L0 choice above used the complete request region to discover a
    // candidate tile.  Re-run the chooser on that child until the descriptor
    // is a fixed point, exactly as the single-matmul planner does.  Otherwise
    // the model could price one L0 geometry while lowering selects another.
    bool stable_output_tile = false;
    for (int iteration = 0; iteration < 8 && !stable_output_tile; ++iteration) {
      int64_t next_m = candidate.output_tile_m;
      int64_t next_n = candidate.output_tile_n;
      try {
        auto init_variant =
            utils::ChooseL0Tile(MakeL0Config(view, hardware, candidate.output_tile_m, candidate.output_tile_n,
                                             one_full_contraction ? node.k : candidate.k_loop.chunk,
                                             /*allow_operand_stationary=*/false, /*c_read=*/false));
        next_m = std::min<int64_t>(next_m, init_variant.m);
        next_n = std::min<int64_t>(next_n, init_variant.n);
        if (!one_full_contraction) {
          auto rolled_variant = utils::ChooseL0Tile(MakeL0Config(
              view, hardware, candidate.output_tile_m, candidate.output_tile_n, candidate.k_loop.chunk,
              /*allow_operand_stationary=*/false, /*c_read=*/true));
          next_m = std::min<int64_t>(next_m, rolled_variant.m);
          next_n = std::min<int64_t>(next_n, rolled_variant.n);
        }
        if (candidate.k_loop.tail > 0) {
          auto tail_variant = utils::ChooseL0Tile(MakeL0Config(
              view, hardware, candidate.output_tile_m, candidate.output_tile_n, candidate.k_loop.tail,
              /*allow_operand_stationary=*/false, /*c_read=*/true));
          next_m = std::min<int64_t>(next_m, tail_variant.m);
          next_n = std::min<int64_t>(next_n, tail_variant.n);
        }
      } catch (const pypto::ValueError&) {
        next_m = 0;
        next_n = 0;
      }
      stable_output_tile = next_m == candidate.output_tile_m && next_n == candidate.output_tile_n;
      candidate.output_tile_m = next_m;
      candidate.output_tile_n = next_n;
      if (candidate.output_tile_m <= 0 || candidate.output_tile_n <= 0) break;
    }
    if (!stable_output_tile || candidate.output_tile_m <= 0 || candidate.output_tile_n <= 0) {
      continue;
    }
    try {
      candidate.l0_init =
          utils::ChooseL0Tile(MakeL0Config(view, hardware, candidate.output_tile_m, candidate.output_tile_n,
                                           one_full_contraction ? node.k : candidate.k_loop.chunk,
                                           /*allow_operand_stationary=*/false, /*c_read=*/false));
      candidate.l0_rolled =
          one_full_contraction
              ? candidate.l0_init
              : utils::ChooseL0Tile(MakeL0Config(view, hardware, candidate.output_tile_m,
                                                 candidate.output_tile_n, candidate.k_loop.chunk,
                                                 /*allow_operand_stationary=*/false, /*c_read=*/true));
      if (candidate.k_loop.tail > 0) {
        candidate.l0_tail = utils::ChooseL0Tile(MakeL0Config(
            view, hardware, candidate.output_tile_m, candidate.output_tile_n, candidate.k_loop.tail,
            /*allow_operand_stationary=*/false, /*c_read=*/true));
      }
    } catch (const pypto::ValueError&) {
      continue;
    }
    candidate.output_tiles_m = CeilDiv(descriptor.output.height, candidate.output_tile_m);
    candidate.output_tiles_n = CeilDiv(descriptor.output.width, candidate.output_tile_n);
    if ((candidate.retain_lhs && candidate.output_tiles_n <= 1) ||
        (candidate.retain_rhs && candidate.output_tiles_m <= 1)) {
      continue;
    }

    const int64_t child_streamed_elements =
        (lhs_cached ? 0 : candidate.output_tile_m) + (rhs_cached ? 0 : candidate.output_tile_n);
    // The first K window is emitted outside the rolled software pipeline. A
    // multi-window request therefore owns one prologue buffer plus one buffer
    // per live rolled stage. Do not assume MemoryReuse happens to coalesce the
    // prologue with a stage: that choice depends on the surrounding serial DAG
    // and previously made lifetime-sharing decisions.
    const int64_t rolled_chunks = candidate.k_loop.full_chunks - 1;
    const int64_t stream_buffers =
        1 + std::min<int64_t>(candidate.k_loop.pipeline_stages, std::max<int64_t>(0, rolled_chunks));
    candidate.peak_transient_l1_bytes =
        retained_bytes + child_streamed_elements * candidate.k_loop.chunk * operand_bytes * stream_buffers;
    if (persistent_l1_bytes + candidate.peak_transient_l1_bytes > hardware.l1_bytes) {
      continue;
    }

    const int64_t child_count = candidate.output_tiles_m * candidate.output_tiles_n;
    const int64_t feed_bytes = child_streamed_elements * candidate.k_loop.chunk * operand_bytes;
    const double feed_cycles = static_cast<double>(feed_bytes) * gm_cycles_per_byte;
    const double init_work =
        L0WorkCycles(view, hardware, candidate.l0_init, candidate.output_tile_m, candidate.output_tile_n,
                     one_full_contraction ? node.k : candidate.k_loop.chunk);
    const double rolled_work =
        one_full_contraction ? init_work
                             : L0WorkCycles(view, hardware, candidate.l0_rolled, candidate.output_tile_m,
                                            candidate.output_tile_n, candidate.k_loop.chunk);
    double child_wall = feed_cycles + init_work;
    if (rolled_chunks > 0) {
      child_wall += KWindowStreamWall(rolled_chunks, candidate.k_loop.pipeline_stages, feed_cycles,
                                      rolled_work, feed_cycles, rolled_work);
    }
    int64_t child_gm_bytes = feed_bytes * candidate.k_loop.full_chunks;
    double child_l0_cycles = init_work + static_cast<double>(candidate.k_loop.full_chunks - 1) * rolled_work;
    if (candidate.k_loop.tail > 0) {
      const int64_t tail_bytes = child_streamed_elements * candidate.k_loop.tail * operand_bytes;
      const double tail_feed = static_cast<double>(tail_bytes) * gm_cycles_per_byte;
      const double tail_work = L0WorkCycles(view, hardware, candidate.l0_tail, candidate.output_tile_m,
                                            candidate.output_tile_n, candidate.k_loop.tail);
      child_wall += tail_feed + tail_work;
      child_gm_bytes += tail_bytes;
      child_l0_cycles += tail_work;
    }
    const double child_drain =
        FinalDrainCycles(view, hardware, candidate.l0_init, candidate.output_tile_m, candidate.output_tile_n);
    child_wall += child_drain;
    candidate.gm_to_l1_bytes = retained_bytes + child_count * child_gm_bytes;
    candidate.modeled_l0_cycles = static_cast<double>(child_count) * child_l0_cycles;
    candidate.modeled_drain_cycles = static_cast<double>(child_count) * child_drain;
    candidate.modeled_cycles = static_cast<double>(retained_bytes) * gm_cycles_per_byte +
                               static_cast<double>(child_count) * child_wall;
    if (!std::isfinite(candidate.modeled_cycles)) continue;
    if (!best || std::tie(candidate.modeled_cycles, candidate.peak_transient_l1_bytes) <
                     std::tie(best->modeled_cycles, best->peak_transient_l1_bytes)) {
      best = std::move(candidate);
    }
  }
  return best;
}

CubeSchedulePlan PlanSerialDag(const CubeGraph& graph, const CubeHardware& hardware) {
  CubeSchedulePlan best;
  const int64_t max_work_units = 2 * static_cast<int64_t>(hardware.cube_cores);
  const auto m_regions = CandidateRegions(graph.m, hardware.fractal, max_work_units);
  const auto n_regions = CandidateRegions(graph.n, hardware.fractal, max_work_units);
  const double gm_cycles_per_byte =
      hardware.core_frequency_hz / (kGiB * hardware.gm_to_l1_bandwidth_gib_per_s);

  for (const auto& [region_m, parts_m] : m_regions) {
    if (parts_m * region_m != graph.m) continue;
    for (const auto& [region_n, parts_n] : n_regions) {
      if (parts_n * region_n != graph.n) continue;
      const int64_t work_units = parts_m * parts_n;
      if (work_units <= 0 || work_units > max_work_units) continue;

      auto requests = BuildRequestDag(graph, region_m, region_n);
      auto residents = DeriveResidentBoundaries(&requests);
      std::vector<size_t> last_use(requests.size(), 0);
      for (size_t index = 0; index < requests.size(); ++index) {
        last_use[index] = index;
      }
      for (size_t index = 0; index < requests.size(); ++index) {
        for (int64_t producer : {requests[index].lhs_producer, requests[index].rhs_producer}) {
          if (producer >= 0) {
            last_use[static_cast<size_t>(producer)] =
                std::max(last_use[static_cast<size_t>(producer)], index);
          }
        }
      }
      std::vector<std::vector<size_t>> producer_releases(requests.size());
      for (size_t producer = 0; producer < requests.size(); ++producer) {
        if (last_use[producer] > producer) {
          producer_releases[last_use[producer]].push_back(producer);
        }
      }
      std::vector<std::vector<size_t>> resident_starts(requests.size());
      std::vector<std::vector<size_t>> resident_releases(requests.size());
      for (const CubeResidentBoundaryPlan& resident : residents) {
        resident_starts[resident.first_use].push_back(resident.id);
        resident_releases[resident.last_use].push_back(resident.id);
      }

      int64_t persistent = 0;
      int64_t peak_l1 = 0;
      int64_t gm_bytes = 0;
      double task_cycles = 0.0;
      bool feasible = true;
      std::vector<int64_t> produced_bytes(requests.size(), 0);
      std::vector<bool> resident_live(residents.size(), false);
      for (size_t index = 0; index < requests.size(); ++index) {
        for (size_t resident_id : resident_starts[index]) {
          const CubeResidentBoundaryPlan& resident = residents[resident_id];
          persistent += resident.bytes;
          resident_live[resident.id] = true;
          gm_bytes += resident.bytes;
          task_cycles += static_cast<double>(resident.bytes) * gm_cycles_per_byte;
        }
        if (!requests[index].is_sink) {
          const CubeMatmulNode& node = graph.matmuls[requests[index].node];
          produced_bytes[index] =
              requests[index].output.height * requests[index].output.width * DTypeBytes(node.storage_dtype);
          persistent += produced_bytes[index];
        }
        auto planned = PlanFixedRequest(graph, requests[index], hardware, persistent);
        if (!planned) {
          feasible = false;
          break;
        }
        requests[index] = std::move(*planned);
        peak_l1 = std::max(peak_l1, persistent + requests[index].peak_transient_l1_bytes);
        gm_bytes += requests[index].gm_to_l1_bytes;
        task_cycles += requests[index].modeled_cycles;

        for (size_t producer : producer_releases[index]) {
          persistent -= produced_bytes[producer];
        }
        for (size_t resident_id : resident_releases[index]) {
          const CubeResidentBoundaryPlan& resident = residents[resident_id];
          if (!resident_live[resident.id]) continue;
          persistent -= resident.bytes;
          resident_live[resident.id] = false;
        }
      }
      if (!feasible || peak_l1 <= 0 || peak_l1 > hardware.l1_bytes) continue;

      CubeSchedulePlan candidate;
      candidate.feasible = true;
      candidate.spatial_policy = CubeSpatialPolicy::Uniform;
      candidate.parts_m = parts_m;
      candidate.parts_n = parts_n;
      candidate.region_m = region_m;
      candidate.region_n = region_n;
      candidate.spatial_work_units = work_units;
      candidate.work_units = work_units;
      candidate.split_k = 1;
      candidate.first_partial_work_units = work_units;
      candidate.atomic_rest_work_units = 0;
      candidate.peak_l1_bytes = peak_l1;
      candidate.gm_to_l1_bytes_per_work_unit = gm_bytes;
      candidate.gm_to_l1_bytes_total = gm_bytes * work_units;
      candidate.modeled_gm_to_l1_cycles = static_cast<double>(gm_bytes) * gm_cycles_per_byte;
      candidate.modeled_split_sync_cycles = 0.0;
      candidate.modeled_cycles = static_cast<double>(CeilDiv(work_units, hardware.cube_cores)) * task_cycles;
      candidate.matmuls = std::move(requests);
      candidate.resident_boundaries = std::move(residents);
      candidate.execution_order.resize(candidate.matmuls.size());
      for (size_t index = 0; index < candidate.execution_order.size(); ++index) {
        candidate.execution_order[index] = index;
        candidate.modeled_l0_cycles += candidate.matmuls[index].modeled_l0_cycles;
        candidate.modeled_final_drain_cycles += candidate.matmuls[index].modeled_drain_cycles;
      }
      const CubeMatmulSchedule& root = candidate.matmuls.back();
      INTERNAL_CHECK(root.is_sink) << "Internal error: serial cube request DAG does not end at its sink";
      candidate.output_tile_m = root.output_tile_m;
      candidate.output_tile_n = root.output_tile_n;
      candidate.output_tiles_m = root.output_tiles_m;
      candidate.output_tiles_n = root.output_tiles_n;
      candidate.k_loop = root.k_loop;
      candidate.retain_lhs = root.retain_lhs;
      candidate.retain_rhs = root.retain_rhs;
      candidate.retained_lhs_bytes = root.retained_lhs_bytes;
      candidate.retained_rhs_bytes = root.retained_rhs_bytes;
      candidate.l0_init = root.l0_init;
      candidate.l0_rolled = root.l0_rolled;
      candidate.l0_tail = root.l0_tail;
      if (!best.feasible ||
          std::tie(candidate.modeled_cycles, candidate.work_units, candidate.peak_l1_bytes) <
              std::tie(best.modeled_cycles, best.work_units, best.peak_l1_bytes)) {
        best = std::move(candidate);
      }
    }
  }
  return best;
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
  if (graph.matmuls.size() > 1) return PlanSerialDag(graph, hardware_);

  const int64_t max_work_units = 2 * static_cast<int64_t>(hardware_.cube_cores);
  const auto m_regions = CandidateRegions(graph.m, hardware_.fractal, max_work_units);
  const auto n_regions = CandidateRegions(graph.n, hardware_.fractal, max_work_units);
  const int64_t operand_bytes = DTypeBytes(graph.operand_dtype);
  const int64_t accumulator_bytes = DTypeBytes(graph.accumulator_dtype);
  const double gm_cycles_per_byte =
      hardware_.core_frequency_hz / (kGiB * hardware_.gm_to_l1_bandwidth_gib_per_s);

  auto better = [](const CubeSchedulePlan& lhs, const CubeSchedulePlan& rhs) {
    if (!rhs.feasible) return true;
    const int lhs_retained = static_cast<int>(lhs.retain_lhs) + static_cast<int>(lhs.retain_rhs);
    const int rhs_retained = static_cast<int>(rhs.retain_lhs) + static_cast<int>(rhs.retain_rhs);
    return std::tie(lhs.modeled_cycles, lhs.work_units, lhs_retained, lhs.region_m, lhs.region_n) <
           std::tie(rhs.modeled_cycles, rhs.work_units, rhs_retained, rhs.region_m, rhs.region_n);
  };

  for (const auto& [region_m, parts_m] : m_regions) {
    for (const auto& [region_n, parts_n] : n_regions) {
      const int64_t spatial_work_units = parts_m * parts_n;
      if (spatial_work_units <= 0 || spatial_work_units > max_work_units) continue;
      const bool uniform = parts_m * region_m == graph.m && parts_n * region_n == graph.n;

      const int64_t max_split = std::min(max_work_units / spatial_work_units, graph.k / hardware_.fractal);
      for (int64_t split_k = 1; split_k <= max_split; ++split_k) {
        // Atomic partials must cover disjoint output tiles and equal K shares.
        // Clamped-overlap remains legal only for the non-atomic split=1 path.
        if (graph.k % split_k != 0 || (split_k > 1 && !uniform)) continue;
        const int64_t effective_k = graph.k / split_k;
        if (effective_k < hardware_.fractal || effective_k % hardware_.fractal != 0) continue;

        const int64_t bytes_per_k = (region_m + region_n) * operand_bytes;
        const int64_t l1_window =
            std::min(effective_k, (hardware_.l1_bytes / bytes_per_k / hardware_.fractal) * hardware_.fractal);
        if (l1_window < hardware_.fractal) continue;

        CubeKLoopPlan k_loop;
        k_loop.l1_window_k = l1_window;
        const int64_t pipelined_chunk = PipelinedChunk(effective_k, l1_window, hardware_.fractal);
        if (pipelined_chunk > 0) {
          k_loop.chunk = pipelined_chunk;
          k_loop.pipeline_stages = 2;
        } else {
          k_loop.chunk = std::max<int64_t>(hardware_.fractal, std::min(l1_window, effective_k));
        }
        k_loop.full_chunks = effective_k / k_loop.chunk;
        k_loop.tail = effective_k - k_loop.full_chunks * k_loop.chunk;
        if (k_loop.chunk <= 0 || k_loop.full_chunks <= 0 || k_loop.tail % hardware_.fractal != 0) {
          continue;
        }

        const bool one_full_contraction = k_loop.full_chunks == 1 && k_loop.tail == 0;
        utils::L0TileResult l0_init;
        utils::L0TileResult l0_rolled;
        utils::L0TileResult l0_tail;
        try {
          l0_init = utils::ChooseL0Tile(MakeL0Config(graph, hardware_, region_m, region_n,
                                                     one_full_contraction ? effective_k : k_loop.chunk,
                                                     /*allow_operand_stationary=*/false,
                                                     /*c_read=*/false));
          l0_rolled = one_full_contraction
                          ? l0_init
                          : utils::ChooseL0Tile(MakeL0Config(graph, hardware_, region_m, region_n,
                                                             k_loop.chunk, /*allow_operand_stationary=*/false,
                                                             /*c_read=*/true));
          if (k_loop.tail > 0) {
            l0_tail = utils::ChooseL0Tile(MakeL0Config(graph, hardware_, region_m, region_n, k_loop.tail,
                                                       /*allow_operand_stationary=*/false, /*c_read=*/true));
          }
        } catch (const pypto::ValueError&) {
          continue;
        }
        if (l0_init.m <= 0 || l0_init.n <= 0 || l0_init.k <= 0 ||
            (k_loop.tail > 0 && (l0_tail.m <= 0 || l0_tail.n <= 0 || l0_tail.k <= 0))) {
          continue;
        }

        // AutoTileMatmulL0 independently replays the one uniform child request
        // emitted inside the serial output-tile loop. Converge on dimensions
        // accepted unchanged by the init, rolled-accumulator, and tail paths.
        // A final child uses a backward-clamped offset rather than a smaller
        // shape, so the request descriptor and cost stay uniform.
        int64_t output_tile_m = std::min<int64_t>(l0_init.m, l0_rolled.m);
        int64_t output_tile_n = std::min<int64_t>(l0_init.n, l0_rolled.n);
        if (k_loop.tail > 0) {
          output_tile_m = std::min<int64_t>(output_tile_m, l0_tail.m);
          output_tile_n = std::min<int64_t>(output_tile_n, l0_tail.n);
        }
        // The first partial and carried accumulated result are distinct SSA
        // identities until semantic accumulator reuse coalesces them.  Budget
        // each identity at half L0C so both the intermediate IR and the final
        // one-buffer algorithm remain buildable.
        const int64_t half_l0c_elements = static_cast<int64_t>(hardware_.l0c_bytes) / (2 * accumulator_bytes);
        while (output_tile_m * output_tile_n > half_l0c_elements) {
          if (output_tile_m >= output_tile_n && output_tile_m > hardware_.fractal) {
            output_tile_m -= hardware_.fractal;
          } else if (output_tile_n > hardware_.fractal) {
            output_tile_n -= hardware_.fractal;
          } else {
            output_tile_m = 0;
            output_tile_n = 0;
            break;
          }
        }
        bool stable_output_tile = false;
        for (int iteration = 0; iteration < 8 && !stable_output_tile; ++iteration) {
          int64_t next_m = output_tile_m;
          int64_t next_n = output_tile_n;
          try {
            auto init_variant =
                utils::ChooseL0Tile(MakeL0Config(graph, hardware_, output_tile_m, output_tile_n,
                                                 one_full_contraction ? effective_k : k_loop.chunk,
                                                 /*allow_operand_stationary=*/false, /*c_read=*/false));
            next_m = std::min<int64_t>(next_m, init_variant.m);
            next_n = std::min<int64_t>(next_n, init_variant.n);
            if (!one_full_contraction) {
              auto rolled_variant = utils::ChooseL0Tile(
                  MakeL0Config(graph, hardware_, output_tile_m, output_tile_n, k_loop.chunk,
                               /*allow_operand_stationary=*/false, /*c_read=*/true));
              next_m = std::min<int64_t>(next_m, rolled_variant.m);
              next_n = std::min<int64_t>(next_n, rolled_variant.n);
            }
            if (k_loop.tail > 0) {
              auto tail_variant = utils::ChooseL0Tile(
                  MakeL0Config(graph, hardware_, output_tile_m, output_tile_n, k_loop.tail,
                               /*allow_operand_stationary=*/false, /*c_read=*/true));
              next_m = std::min<int64_t>(next_m, tail_variant.m);
              next_n = std::min<int64_t>(next_n, tail_variant.n);
            }
          } catch (const pypto::ValueError&) {
            next_m = 0;
            next_n = 0;
          }
          stable_output_tile = next_m == output_tile_m && next_n == output_tile_n;
          output_tile_m = next_m;
          output_tile_n = next_n;
          if (output_tile_m <= 0 || output_tile_n <= 0) break;
        }
        if (!stable_output_tile || output_tile_m <= 0 || output_tile_n <= 0) continue;

        try {
          l0_init = utils::ChooseL0Tile(MakeL0Config(graph, hardware_, output_tile_m, output_tile_n,
                                                     one_full_contraction ? effective_k : k_loop.chunk,
                                                     /*allow_operand_stationary=*/false, /*c_read=*/false));
          l0_rolled = one_full_contraction ? l0_init
                                           : utils::ChooseL0Tile(MakeL0Config(
                                                 graph, hardware_, output_tile_m, output_tile_n, k_loop.chunk,
                                                 /*allow_operand_stationary=*/false, /*c_read=*/true));
          if (k_loop.tail > 0) {
            l0_tail =
                utils::ChooseL0Tile(MakeL0Config(graph, hardware_, output_tile_m, output_tile_n, k_loop.tail,
                                                 /*allow_operand_stationary=*/false, /*c_read=*/true));
          }
        } catch (const pypto::ValueError&) {
          continue;
        }
        const int64_t output_tiles_m = CeilDiv(region_m, output_tile_m);
        const int64_t output_tiles_n = CeilDiv(region_n, output_tile_n);
        if (output_tile_m <= 0 || output_tile_n <= 0 || output_tiles_m <= 0 || output_tiles_n <= 0) {
          continue;
        }

        // Retention is useful only when a boundary panel feeds more than one
        // child output tile. It extends the panel lifetime across that exact
        // serial tile loop; ordinary SSA liveness releases it after
        // the last child.
        for (int retained_mask = 0; retained_mask < 4; ++retained_mask) {
          const bool retain_lhs = (retained_mask & 1) != 0;
          const bool retain_rhs = (retained_mask & 2) != 0;
          if ((retain_lhs && output_tiles_n <= 1) || (retain_rhs && output_tiles_m <= 1)) continue;

          CubeKLoopPlan candidate_k_loop = k_loop;
          if (retain_lhs && retain_rhs) candidate_k_loop.pipeline_stages = 1;
          const int64_t retained_lhs_bytes = retain_lhs ? region_m * effective_k * operand_bytes : 0;
          const int64_t retained_rhs_bytes = retain_rhs ? effective_k * region_n * operand_bytes : 0;
          const int64_t streamed_elements_per_stage =
              (retain_lhs ? 0 : output_tile_m) + (retain_rhs ? 0 : output_tile_n);
          const int64_t peak_l1_bytes = retained_lhs_bytes + retained_rhs_bytes +
                                        streamed_elements_per_stage * candidate_k_loop.chunk * operand_bytes *
                                            candidate_k_loop.pipeline_stages;
          if (peak_l1_bytes <= 0 || peak_l1_bytes > hardware_.l1_bytes) continue;

          int64_t gm_bytes = retained_lhs_bytes + retained_rhs_bytes;
          double task_wall = static_cast<double>(gm_bytes) * gm_cycles_per_byte;
          double l0_cycles = 0.0;
          double final_drain_cycles = 0.0;
          const int64_t child_count = output_tiles_m * output_tiles_n;
          const int64_t feed_elements = (retain_lhs ? 0 : output_tile_m) + (retain_rhs ? 0 : output_tile_n);
          const int64_t feed_bytes = feed_elements * candidate_k_loop.chunk * operand_bytes;
          const double feed_cycles = static_cast<double>(feed_bytes) * gm_cycles_per_byte;
          const double init_work = L0WorkCycles(graph, hardware_, l0_init, output_tile_m, output_tile_n,
                                                one_full_contraction ? effective_k : candidate_k_loop.chunk);
          const double rolled_work = one_full_contraction
                                         ? init_work
                                         : L0WorkCycles(graph, hardware_, l0_rolled, output_tile_m,
                                                        output_tile_n, candidate_k_loop.chunk);
          double child_wall = feed_cycles + init_work;
          const int64_t rolled_chunks = candidate_k_loop.full_chunks - 1;
          if (rolled_chunks > 0) {
            child_wall += KWindowStreamWall(rolled_chunks, candidate_k_loop.pipeline_stages, feed_cycles,
                                            rolled_work, feed_cycles, rolled_work);
          }
          int64_t child_gm_bytes = feed_bytes * candidate_k_loop.full_chunks;
          double child_l0_cycles =
              init_work + static_cast<double>(candidate_k_loop.full_chunks - 1) * rolled_work;
          if (candidate_k_loop.tail > 0) {
            const int64_t tail_bytes = feed_elements * candidate_k_loop.tail * operand_bytes;
            const double tail_feed = static_cast<double>(tail_bytes) * gm_cycles_per_byte;
            const double tail_work =
                L0WorkCycles(graph, hardware_, l0_tail, output_tile_m, output_tile_n, candidate_k_loop.tail);
            child_wall += tail_feed + tail_work;
            child_gm_bytes += tail_bytes;
            child_l0_cycles += tail_work;
          }
          const double child_drain =
              FinalDrainCycles(graph, hardware_, l0_init, output_tile_m, output_tile_n);
          child_wall += child_drain;
          task_wall += static_cast<double>(child_count) * child_wall;
          gm_bytes += child_count * child_gm_bytes;
          l0_cycles += static_cast<double>(child_count) * child_l0_cycles;
          final_drain_cycles += static_cast<double>(child_count) * child_drain;

          CubeSchedulePlan candidate;
          candidate.feasible = true;
          candidate.parts_m = parts_m;
          candidate.parts_n = parts_n;
          candidate.region_m = region_m;
          candidate.region_n = region_n;
          candidate.output_tile_m = output_tile_m;
          candidate.output_tile_n = output_tile_n;
          candidate.output_tiles_m = output_tiles_m;
          candidate.output_tiles_n = output_tiles_n;
          candidate.spatial_work_units = spatial_work_units;
          candidate.split_k = split_k;
          candidate.first_partial_work_units = spatial_work_units;
          candidate.atomic_rest_work_units = spatial_work_units * (split_k - 1);
          candidate.work_units = spatial_work_units * split_k;
          candidate.spatial_policy = uniform ? CubeSpatialPolicy::Uniform : CubeSpatialPolicy::ClampedOverlap;
          candidate.k_loop = candidate_k_loop;
          candidate.peak_l1_bytes = peak_l1_bytes;
          candidate.gm_to_l1_bytes_per_work_unit = gm_bytes;
          candidate.gm_to_l1_bytes_total = gm_bytes * candidate.work_units;
          candidate.retain_lhs = retain_lhs;
          candidate.retain_rhs = retain_rhs;
          candidate.retained_lhs_bytes = retained_lhs_bytes;
          candidate.retained_rhs_bytes = retained_rhs_bytes;
          candidate.modeled_gm_to_l1_cycles = static_cast<double>(gm_bytes) * gm_cycles_per_byte;
          candidate.modeled_l0_cycles = l0_cycles;
          candidate.modeled_final_drain_cycles = final_drain_cycles;
          // The two SPMD phases are submitted in program order. Device evidence
          // supports no additional transferable synchronization cost beyond the
          // phase boundary, so keep this explicit term at zero until a stable
          // silicon coefficient is available.
          candidate.modeled_split_sync_cycles = 0.0;
          const int64_t first_waves = CeilDiv(candidate.first_partial_work_units, hardware_.cube_cores);
          const int64_t rest_waves = candidate.atomic_rest_work_units == 0
                                         ? 0
                                         : CeilDiv(candidate.atomic_rest_work_units, hardware_.cube_cores);
          candidate.modeled_cycles =
              static_cast<double>(first_waves + rest_waves) * task_wall + candidate.modeled_split_sync_cycles;
          candidate.l0_init = l0_init;
          candidate.l0_rolled = l0_rolled;
          candidate.l0_tail = l0_tail;
          if (std::isfinite(candidate.modeled_cycles) && better(candidate, best)) best = candidate;
        }
      }
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
