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

#ifndef SRC_IR_TRANSFORMS_UTILS_SCOPED_SYNC_PALETTE_H_
#define SRC_IR_TRANSFORMS_UTILS_SCOPED_SYNC_PALETTE_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "src/ir/transforms/utils/scoped_token_order.h"

namespace pypto::ir::sync {
// Original normalized-coordinate progression, not a new occurrence identity.
struct ScopedPublicationDigit {
  int64_t first = 0, step = 1, count = 1;
};
struct ScopedPublicationRankPiece {
  size_t raw_use = 0;
  int64_t rank_base = 0;
  std::vector<ScopedPublicationDigit> source, target;
  // Coefficients multiply the progression indices in source, not raw IVs.
  std::vector<int64_t> rank_coefficients;
};
struct ScopedPaletteWork {
  size_t grammar_nodes = 0, phase_visits = 0, domain_visits = 0;
  size_t rank_pieces = 0, transition_queries = 0;
  size_t routing_entries = 0, boundary_node_visits = 0, proof_axis_entries = 0;
};
class ScopedPublicationSchedule {
 public:
  const std::shared_ptr<const ScopedSyncProtocol>& Protocol() const;
  PipeType SourcePipe() const;
  PipeType TargetPipe() const;
  int64_t Publications() const;
  const std::vector<ScopedPublicationRankPiece>& Pieces() const;

 private:
  struct State;
  std::shared_ptr<const State> state_;
  explicit ScopedPublicationSchedule(std::shared_ptr<const State> state);
  friend struct ScopedPaletteBuilder;
};
class ScopedPaletteCertificate {
 public:
  const std::shared_ptr<const ScopedPublicationSchedule>& Schedule() const { return schedule_; }
  int64_t Budget() const { return budget_; }

 private:
  std::shared_ptr<const ScopedPublicationSchedule> schedule_;
  int64_t budget_;
  ScopedPaletteCertificate(std::shared_ptr<const ScopedPublicationSchedule> schedule, int64_t budget)
      : schedule_(std::move(schedule)), budget_(budget) {}
  friend struct ScopedPaletteBuilder;
};
enum class ScopedPaletteStatus {
  Certified,
  UnsupportedGrammar,
  ArithmeticOverflow,
  UnprovedReuse,
  Constructed
};
const char* ScopedPaletteStatusName(ScopedPaletteStatus status);
struct ScopedPaletteResult {
  ScopedPaletteStatus status = ScopedPaletteStatus::UnsupportedGrammar;
  std::string reason;
  std::shared_ptr<const ScopedPublicationSchedule> schedule;
  std::shared_ptr<const ScopedPaletteCertificate> certificate;
  ScopedPaletteWork work;
  size_t attempted_budgets = 0;
};
// Production construction and certification. B comes from the existing resource owner;
// failure proves no capacity lower bound. No queries become token graph edges.
// Counts and ranks are checked int64; overflowing models remain unavailable.
// Work is charged to original-tree depth, exact phase partitions, raw uses,
// rank pieces and B bounded boundary queries. LCM phase expansion is explicit;
// this does not claim a trip-independent bound when the supplied phase grows.
ScopedPaletteResult BuildScopedPublicationSchedule(const ScopedTokenOrder& order, PipeType source,
                                                   PipeType target);
ScopedPaletteResult CertifyScopedPalette(const ScopedTokenOrder& order,
                                         std::shared_ptr<const ScopedPublicationSchedule> schedule,
                                         int64_t budget);
}  // namespace pypto::ir::sync
#endif  // SRC_IR_TRANSFORMS_UTILS_SCOPED_SYNC_PALETTE_H_
