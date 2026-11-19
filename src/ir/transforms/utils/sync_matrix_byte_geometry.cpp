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

#include "src/ir/transforms/utils/sync_matrix_byte_geometry.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_config.h"
#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/core/logging.h"
#include "pypto/ir/transforms/pass_context.h"

namespace pypto::ir::sync {
bool IsSyncMatrixSpace(MemorySpace space) {
  return space == MemorySpace::Mat || space == MemorySpace::Left || space == MemorySpace::Right ||
         space == MemorySpace::Acc;
}

namespace {

// Limit exact striped output per static effect. Whole inner-axis spans are
// contiguous and use one run regardless of tile width. No loop visits or scalar
// elements are expanded. Storage normalization sorts its N descriptors once
// and then does O(N + R) work, where R counts original effects and emitted runs.
constexpr int64_t kMaxByteRunsPerEffect = 128;
constexpr int64_t kFractalAxis = 16;
constexpr int64_t kFractalRowBytes = 32;
constexpr int64_t kStandardFractalBytes = 512;
constexpr int64_t kAccumulatorFractalBytes = 1024;

int64_t CheckedByteCount(__int128 value) {
  CHECK(value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Overlapping placed buffers require representable byte geometry";
  return static_cast<int64_t>(value);
}

struct ByteLayout {
  int64_t box_rows, box_columns, box_bytes;
  bool columns_outer;
};

ByteLayout QualifyByteLayout(const SyncStorage& storage) {
  const auto* context = PassContext::Current();
  const auto* target = context ? context->GetBackend() : backend::GetBackend();
  const auto alignment = target ? target->GetMemAlignment(storage.space) : 0;
  const auto& type = storage.layout;
  CHECK(storage.slot_count == 1 && alignment && storage.address >= 0 &&
        static_cast<uint64_t>(storage.address) % alignment == 0 && type && storage.rows > 0 &&
        storage.columns > 0 && type->valid_shape_ == type->shape_ && type->compact_ == CompactMode::null &&
        type->pad_ == PadValue::null)
      << "Overlapping placed buffers need an aligned, fully valid single-slot partition";
  CHECK(type->blayout_ == TileLayout::col_major || type->blayout_ == TileLayout::row_major)
      << "Overlapping placed buffers have an unsupported outer layout";
  const auto bytes = CheckedByteCount(type->dtype_.GetByte());
  CHECK(bytes > 0 && type->dtype_.GetBit() % 8 == 0) << "Overlapping placed buffers need whole-byte elements";
  const bool columns_outer = type->blayout_ == TileLayout::col_major;
  if (type->slayout_ == TileLayout::none_box) {
    return {1, 1, bytes, columns_outer};
  }
  const bool inner_columns = type->slayout_ == TileLayout::col_major;
  CHECK(type->slayout_ == TileLayout::row_major || (inner_columns && !columns_outer))
      << "Overlapping placed buffers require a native NZ, ZN or ZZ layout";
  const auto fractal = type->fractal_;
  CHECK((fractal == kStandardFractalBytes && kFractalRowBytes % bytes == 0) ||
        (fractal == kAccumulatorFractalBytes && bytes == 4))
      << "Overlapping placed buffers require an accounted native fractal size";
  const auto minor = fractal == kAccumulatorFractalBytes ? kFractalAxis : kFractalRowBytes / bytes;
  const auto rows = inner_columns ? minor : kFractalAxis;
  const auto cols = inner_columns ? kFractalAxis : minor;
  CHECK(storage.rows % rows == 0 && storage.columns % cols == 0)
      << "Overlapping placed buffers require fully tiled physical layouts";
  return {rows, cols, rows * cols * bytes, columns_outer};
}

struct Axis {
  int64_t first, step, count;
};

Axis InBoxes(int64_t first, int64_t step, int64_t count, int64_t box) {
  CHECK(first >= 0 && count > 0 && step >= 0 && first % box == 0 && count % box == 0 && step % box == 0)
      << "Overlapping placed buffers need whole-box access offsets, extents, and steps";
  return {first / box, step / box, count / box};
}

void AppendByteRuns(const SyncAccess& access, const SyncStorage& original, const ByteLayout& layout,
                    size_t storage, int64_t group_address, std::vector<SyncAccess>& output) {
  CHECK(access.exact_footprint && !access.external && !access.slot_coordinate_frame &&
        access.slot_first == 0 && access.slot_step == 0)
      << "Overlapping placed buffers need exact single-slot accesses";
  if (original.space == MemorySpace::Acc) {
    CHECK(access.row == 0 && access.column == 0 && access.row_step == 0 && access.column_step == 0 &&
          access.rows == original.rows && access.columns == original.columns)
        << "Overlapping Acc buffers require whole-window accesses";
  }
  const auto row = InBoxes(access.row, access.row_step, access.rows, layout.box_rows);
  const auto col = InBoxes(access.column, access.column_step, access.columns, layout.box_columns);
  const auto [major, minor] = layout.columns_outer ? std::pair{col, row} : std::pair{row, col};
  const int64_t pitch =
      layout.columns_outer ? original.rows / layout.box_rows : original.columns / layout.box_columns;
  const bool contiguous = minor.count == pitch;
  const int64_t runs = contiguous ? 1 : major.count;
  CHECK(runs <= kMaxByteRunsPerEffect)
      << "Overlapping placed buffers exceed the exact physical-byte run limit";
  const auto box_index = [&](int64_t major_digit, int64_t minor_digit) {
    return CheckedByteCount(__int128(major_digit) * pitch + minor_digit);
  };
  const auto start = CheckedByteCount(__int128(original.address - group_address) +
                                      __int128(box_index(major.first, minor.first)) * layout.box_bytes);
  const auto step = CheckedByteCount(__int128(box_index(major.step, minor.step)) * layout.box_bytes);
  const auto pitch_bytes = CheckedByteCount(__int128(pitch) * layout.box_bytes);
  const auto run_bytes = CheckedByteCount(__int128(minor.count) * layout.box_bytes);
  for (int64_t run = 0; run < runs; ++run) {
    auto bytes = access;
    bytes.storage = storage;
    bytes.row = 0;
    bytes.row_step = 0;
    bytes.rows = 1;
    bytes.column = CheckedByteCount(__int128(start) + __int128(run) * pitch_bytes);
    bytes.column_step = step;
    bytes.columns = contiguous ? CheckedByteCount(__int128(major.count) * run_bytes) : run_bytes;
    output.push_back(std::move(bytes));
  }
}

struct OverlapGroup {
  size_t first_storage;
  int64_t address, end;
  std::optional<size_t> canonical_storage;
};

struct StorageMapping {
  size_t canonical = 0;
  std::optional<size_t> group;
  std::optional<ByteLayout> layout;
};

std::vector<OverlapGroup> FindOverlaps(const std::vector<SyncStorage>& storage,
                                       std::vector<StorageMapping>& mapping) {
  std::vector<size_t> sorted;
  for (size_t i = 0; i < storage.size(); ++i) {
    if (IsSyncMatrixSpace(storage[i].space)) {
      sorted.push_back(i);
    }
  }
  std::sort(sorted.begin(), sorted.end(), [&](size_t a, size_t b) {
    return std::tie(storage[a].space, storage[a].address, a) <
           std::tie(storage[b].space, storage[b].address, b);
  });
  std::vector<OverlapGroup> groups;
  for (size_t begin = 0; begin < sorted.size();) {
    const auto first = sorted[begin];
    const auto& head = storage[first];
    int64_t end = head.address + head.bytes;
    int64_t common_start = head.address, common_end = end;
    size_t after = begin + 1;
    for (; after < sorted.size(); ++after) {
      const auto& next = storage[sorted[after]];
      if (next.space != head.space || next.address >= end) {
        break;
      }
      const auto next_end = next.address + next.bytes;
      end = std::max(end, next_end);
      common_start = next.address;
      common_end = std::min(common_end, next_end);
    }
    if (after - begin > 1) {
      // Accumulator continuation has one interference counter per domain.
      // Every whole-window Acc effect must really overlap every native view
      // in that domain. A merely transitive component cannot prove this.
      CHECK(head.space != MemorySpace::Acc || common_start < common_end)
          << "Overlapping Acc buffers require pairwise-intersecting whole windows";
      for (size_t i = begin; i < after; ++i) {
        auto& entry = mapping[sorted[i]];
        entry.group = groups.size();
        entry.layout = QualifyByteLayout(storage[sorted[i]]);
      }
      groups.push_back({first, head.address, end, std::nullopt});
    }
    begin = after;
  }
  return groups;
}

std::vector<SyncStorage> BuildCanonicalStorage(const std::vector<SyncStorage>& original,
                                               std::vector<StorageMapping>& mapping,
                                               std::vector<OverlapGroup>& groups) {
  std::vector<SyncStorage> storage;
  storage.reserve(original.size());
  for (size_t i = 0; i < original.size(); ++i) {
    auto& entry = mapping[i];
    if (!entry.group) {
      entry.canonical = storage.size();
      storage.push_back(original[i]);
      continue;
    }
    auto& group = groups[*entry.group];
    if (!group.canonical_storage) {
      group.canonical_storage = storage.size();
      const auto& view = original[group.first_storage];
      const auto bytes = group.end - group.address;
      storage.push_back(
          {view.name + " physical bytes", view.space, group.address, 1, bytes, bytes, nullptr, true});
    }
    entry.canonical = *group.canonical_storage;
  }
  return storage;
}

void RewriteSiteStorage(SyncSite& site, const std::vector<SyncStorage>& original,
                        const std::vector<StorageMapping>& mapping, const std::vector<OverlapGroup>& groups) {
  if (site.protected_accumulator) {
    auto& binding = *site.protected_accumulator;
    binding.storage = mapping.at(binding.storage).canonical;
  }
  std::vector<SyncAccess> accesses;
  accesses.reserve(site.accesses.size());
  for (auto& access : site.accesses) {
    const auto& entry = mapping.at(access.storage);
    if (entry.layout) {
      AppendByteRuns(access, original.at(access.storage), *entry.layout, entry.canonical,
                     groups.at(*entry.group).address, accesses);
    } else {
      access.storage = entry.canonical;
      accesses.push_back(std::move(access));
    }
  }
  site.accesses = std::move(accesses);
}

}  // namespace

void CanonicalizeMatrixSyncStorage(CompactSyncInput& input) {
  std::vector<StorageMapping> mapping(input.storage.size());
  auto groups = FindOverlaps(input.storage, mapping);
  if (groups.empty()) {
    return;
  }
  backend::RequireLocalSyncExecutionContract(input.hardware_target);
  auto storage = BuildCanonicalStorage(input.storage, mapping, groups);
  for (auto& segment : input.segments) {
    for (auto& site : segment.sites) {
      RewriteSiteStorage(site, input.storage, mapping, groups);
    }
  }
  input.storage = std::move(storage);
}
}  // namespace pypto::ir::sync
