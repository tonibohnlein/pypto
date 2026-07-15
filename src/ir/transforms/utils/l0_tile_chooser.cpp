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

#include "pypto/ir/transforms/utils/l0_tile_chooser.h"

#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>

#include "core/l0_matmul_plan.h"
#include "pypto/core/logging.h"

namespace pypto {
namespace ir {
namespace utils {

namespace {

Stationarity FromFuseboxStationarity(L0Stationarity stationarity) {
  switch (stationarity) {
    case L0Stationarity::Output:
      return Stationarity::kOutputStationary;
    case L0Stationarity::A:
      return Stationarity::kAStationary;
    case L0Stationarity::B:
      return Stationarity::kBStationary;
  }
  INTERNAL_UNREACHABLE << "Internal error: unknown PTO Fusebox L0 stationarity";
}

}  // namespace

std::string EncodeL0MatmulPlanRecord(const L0MatmulPlanRecord& record) {
  const std::array<int64_t, 23> fields = {
      L0MatmulPlanRecord::kVersion,
      record.source_m,
      record.source_n,
      record.source_k,
      record.bytes_a,
      record.bytes_b,
      record.bytes_c,
      record.accumulator_read ? 1 : 0,
      static_cast<int64_t>(record.output_target),
      record.tile_m,
      record.tile_n,
      record.tile_k,
      static_cast<int64_t>(record.stationarity),
      record.output_stationary_holds_a ? 1 : 0,
      record.buffer_depth_a,
      record.buffer_depth_b,
      record.buffer_depth_c,
      record.k_full_chunks,
      record.k_tail,
      record.k_pipeline_stages,
      record.estimated_traffic_bytes,
      record.estimated_cost_cycles,
      record.padded_compute_volume,
  };
  std::string encoded;
  for (size_t i = 0; i < fields.size(); ++i) {
    if (i != 0) encoded.push_back(',');
    encoded += std::to_string(fields[i]);
  }
  return encoded;
}

std::optional<L0MatmulPlanRecord> DecodeL0MatmulPlanRecord(const std::string& encoded) {
  std::array<int64_t, 23> fields{};
  std::string_view remaining(encoded);
  for (size_t i = 0; i < fields.size(); ++i) {
    const size_t comma = remaining.find(',');
    const std::string_view token = remaining.substr(0, comma);
    if (token.empty()) return std::nullopt;
    const char* begin = token.data();
    const char* end = token.data() + token.size();
    auto [ptr, ec] = std::from_chars(begin, end, fields[i]);
    if (ec != std::errc{} || ptr != end) return std::nullopt;
    if (i + 1 == fields.size()) {
      if (comma != std::string_view::npos) return std::nullopt;
    } else {
      if (comma == std::string_view::npos) return std::nullopt;
      remaining.remove_prefix(comma + 1);
    }
  }
  if (fields[0] != L0MatmulPlanRecord::kVersion) return std::nullopt;
  if ((fields[7] != 0 && fields[7] != 1) || fields[8] < static_cast<int64_t>(L0PlanOutputTarget::kAcc) ||
      fields[8] > static_cast<int64_t>(L0PlanOutputTarget::kL1) ||
      fields[12] < static_cast<int64_t>(Stationarity::kOutputStationary) ||
      fields[12] > static_cast<int64_t>(Stationarity::kBStationary) || (fields[13] != 0 && fields[13] != 1) ||
      fields[17] < 0 || fields[18] < 0 || (fields[19] != 1 && fields[19] != 2)) {
    return std::nullopt;
  }

  L0MatmulPlanRecord record;
  record.source_m = fields[1];
  record.source_n = fields[2];
  record.source_k = fields[3];
  record.bytes_a = fields[4];
  record.bytes_b = fields[5];
  record.bytes_c = fields[6];
  record.accumulator_read = fields[7] != 0;
  record.output_target = static_cast<L0PlanOutputTarget>(fields[8]);
  record.tile_m = fields[9];
  record.tile_n = fields[10];
  record.tile_k = fields[11];
  record.stationarity = static_cast<Stationarity>(fields[12]);
  record.output_stationary_holds_a = fields[13] != 0;
  record.buffer_depth_a = fields[14];
  record.buffer_depth_b = fields[15];
  record.buffer_depth_c = fields[16];
  record.k_full_chunks = fields[17];
  record.k_tail = fields[18];
  record.k_pipeline_stages = fields[19];
  record.estimated_traffic_bytes = fields[20];
  record.estimated_cost_cycles = fields[21];
  record.padded_compute_volume = fields[22];
  return record;
}

L0TileResult ChooseL0Tile(const L0TileConfig& cfg) {
  L0MatmulConfig shared;
  shared.m = cfg.M;
  shared.n = cfg.N;
  shared.k = cfg.K;
  shared.l0a_bytes = cfg.l0a_bytes;
  shared.l0b_bytes = cfg.l0b_bytes;
  shared.l0c_bytes = cfg.l0c_bytes;
  shared.bytes_a = cfg.bytes_a;
  shared.bytes_b = cfg.bytes_b;
  shared.bytes_c = cfg.bytes_c;
  shared.min_m = cfg.min_m;
  shared.min_n = cfg.min_n;
  shared.min_k = cfg.min_k;
  shared.align_m = cfg.align_m;
  shared.align_n = cfg.align_n;
  shared.align_k = cfg.align_k;
  shared.allow_a_stationary = cfg.allow_a_stationary;
  shared.allow_b_stationary = cfg.allow_b_stationary;
  shared.allow_double_buffer_c = cfg.allow_double_buffer_c;
  shared.accumulator_read = cfg.c_read;
  switch (cfg.output_target) {
    case L0PlanOutputTarget::kAcc:
      shared.output_target = L0OutputTarget::Acc;
      break;
    case L0PlanOutputTarget::kGM:
      shared.output_target = L0OutputTarget::GM;
      break;
    case L0PlanOutputTarget::kL1:
      shared.output_target = L0OutputTarget::L1;
      break;
  }
  shared.allow_padding = cfg.allow_padding;
  shared.allow_k_boundary = cfg.allow_k_boundary;
  shared.bw_l0a = cfg.bw_a;
  shared.bw_l0b = cfg.bw_b;
  shared.bw_drain = cfg.bw_drain;
  shared.drain_fixed_cycles = cfg.drain_fixed_cycles;
  shared.drain_row_cycles = cfg.drain_row_cycles;
  shared.drain_penalty_cycles = cfg.drain_penalty_cycles;
  shared.drain_c0_bytes = cfg.drain_c0_bytes;
  shared.mad_head_cycles = cfg.mad_head;
  shared.mad_k_fractal_bytes = cfg.mad_k_fractal_bytes;

  const L0MatmulPlan plan = choose_l0_matmul_plan(shared);
  CHECK(plan.feasible) << "ChooseL0Tile: " << plan.diagnostic;

  L0TileResult result;
  result.m = static_cast<int>(plan.m);
  result.n = static_cast<int>(plan.n);
  result.k = static_cast<int>(plan.k);
  result.estimated_traffic_bytes = plan.estimated_traffic_bytes;
  result.estimated_cost_cycles = plan.estimated_cost_cycles;
  result.padded_compute_volume = plan.padded_compute_volume;
  result.stationarity = FromFuseboxStationarity(plan.stationarity);
  result.os_holds_a = plan.output_stationary_holds_a;
  result.double_buffer_c = plan.buffer_depth_c == 2;
  result.perf_hint = plan.diagnostic;
  return result;
}

}  // namespace utils
}  // namespace ir
}  // namespace pypto
