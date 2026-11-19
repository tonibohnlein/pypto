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

#ifndef SRC_IR_TRANSFORMS_UTILS_SYNC_LOOP_TRIP_PROOF_H_
#define SRC_IR_TRANSFORMS_UTILS_SYNC_LOOP_TRIP_PROOF_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "pypto/ir/expr.h"
#include "pypto/ir/function.h"
#include "pypto/ir/stmt.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync {

// Symbolic integer range + optional affine/modular decomposition wrt one enclosing
// sequential ForStmt `affine_frame`.
struct SymbolicIntRange {
  int64_t lo = 0;
  int64_t hi = 0;
  ExprPtr expr;
  // Optional affine decomposition wrt one enclosing sequential ForStmt `affine_frame`:
  // - If modulus == 0: value = (symbolic_base ? symbolic_base : 0) + affine_base + affine_coeff * iv_norm
  // - If modulus > 0:  value = (affine_base + affine_coeff * iv_norm) mod modulus
  // where iv_norm = (iv - frame_start) / frame_step in [0, frame_trips).
  const ForStmt* affine_frame = nullptr;
  int64_t affine_coeff = 0;
  int64_t affine_base = 0;
  int64_t modulus = 0;
  ExprPtr symbolic_base;

  bool is_const() const { return lo == hi; }
};

// Shared O(N) non-wrapping __int128 range and trip-count prover for synchronization
// consumers (SyncFootprintResolver, ConservativeSyncDataflow, ClosedPrefixSync).
//
// Soundness invariants:
// - Correlated difference cancellation (e.g. (base + 64) - base) is strictly gated
//   on __int128 non-overflow of every arithmetic subexpression in start_ and stop_
//   under proved bounds.
// - Unbounded formal scalar parameters receive NO synthetic > 0 assumption.
class SyncControlRangeProver {
 public:
  explicit SyncControlRangeProver(const FunctionPtr& function, const SyncParameterBounds& parameters = {},
                                  const std::optional<std::pair<int64_t, int64_t>>& blocks = std::nullopt);
  ~SyncControlRangeProver();
  SyncControlRangeProver(SyncControlRangeProver&&) noexcept;
  SyncControlRangeProver& operator=(SyncControlRangeProver&&) noexcept;
  SyncControlRangeProver(const SyncControlRangeProver&) = delete;
  SyncControlRangeProver& operator=(const SyncControlRangeProver&) = delete;

  // True iff `loop` is proved to execute >= 1 iteration on every entry.
  bool IsPositiveTrip(const ForStmt* loop) const;
  // True iff `loop` is proved to execute <= 1 iteration on every entry (0 or 1 trip).
  bool IsAtMostOneTrip(const ForStmt* loop) const;

  // Matches `loop` directly or, when `loop` is a realization clone, an original
  // ForStmt with identical loop_var_ and structurally equal bounds.
  bool IsPositiveTripCompatible(const ForStmt* loop) const;

  // Query bounds + affine decomposition of `expr` under the dominating constraints
  // at `cut` (or function-level bounds if `cut == nullptr`).
  std::optional<SymbolicIntRange> LookupAtCut(const Stmt* cut, const ExprPtr& expr) const;

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace pypto::ir::sync

#endif  // SRC_IR_TRANSFORMS_UTILS_SYNC_LOOP_TRIP_PROOF_H_
