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

#ifndef SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_REPLAY_H_
#define SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_REPLAY_H_

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <tuple>

#include "pypto/ir/transforms/utils/compact_sync_analysis.h"

namespace pypto::ir::sync::replay {

using Ranks = std::array<size_t, kPipeCount>;
enum class StepKind { Barrier, Wait, Payload, Set };

struct Step {
  StepKind kind;
  SyncDirection direction;
  int id = 0;
};

inline void IncludeRanks(Ranks& dst, const Ranks& src) {
  for (size_t p = 0; p < kPipeCount; ++p) {
    dst[p] = std::max(dst[p], src[p]);
  }
}

struct Prefix {
  Ranks payload{};
  Ranks wait{};

  void Include(const Prefix& other) {
    IncludeRanks(payload, other.payload);
    IncludeRanks(wait, other.wait);
  }
};

struct Bank {
  std::optional<Prefix> published;
  size_t last_wait = 0;
};

// SET snapshots completion, WAIT gates future issue, and a previous WAIT is
// a separate fact from completion of any payload. Emitting a SET does not
// gate the following payload on its source.
class QueryQueues {
 public:
  explicit QueryQueues(bool scalar_serial) : scalar_serial_(scalar_serial) {}

  size_t Cut() const { return cut_; }
  // A deletion proof keeps original tape positions so ranks remain comparable.
  void Skip() { ++cut_; }
  const Ranks& IssuePrefix(PipeType pipe) const { return gates_[pipe].payload; }

  bool Apply(const Step& step, const Ranks* required = nullptr, size_t base = 0, const Ranks* older = nullptr,
             size_t older_base = 0) {
    ++cut_;
    const auto [source, target] = step.direction;
    if (step.kind == StepKind::Payload) {
      auto seen = gates_[target];
      if ((required && !Covers(seen.payload, *required, base)) ||
          (older && !Covers(seen.payload, *older, older_base))) {
        return false;
      }
      outstanding_[target] = cut_;
      if (target == PipeType::S && scalar_serial_) {
        seen.payload[target] = cut_;
        gates_[target] = seen;
      }
      return true;
    }
    if (step.kind == StepKind::Barrier) {
      gates_[target] = commands_[target] = Capture(target);
      return true;
    }
    auto& bank = banks_[{source, target, step.id}];
    if (step.kind == StepKind::Set) {
      auto captured = Capture(source);
      if (bank.published || captured.wait[target] < bank.last_wait) {
        return false;
      }
      bank.published = commands_[source] = captured;
      return true;
    }
    if (!bank.published) {
      return false;
    }
    auto consumed = gates_[target];
    consumed.Include(*bank.published);
    bank.published.reset();
    consumed.wait[target] = bank.last_wait = cut_;
    gates_[target] = consumed;
    commands_[target].Include(consumed);
    return true;
  }

  bool AllConsumed() const {
    return std::all_of(banks_.begin(), banks_.end(), [](const auto& bank) { return !bank.second.published; });
  }

 private:
  static bool Covers(const Ranks& actual, const Ranks& required, size_t base) {
    for (size_t p = 0; p < kPipeCount; ++p) {
      if (required[p] && actual[p] < base + required[p]) {
        return false;
      }
    }
    return true;
  }

  Prefix Capture(PipeType pipe) const {
    auto prefix = gates_[pipe];
    prefix.Include(commands_[pipe]);
    prefix.payload[pipe] = std::max(prefix.payload[pipe], outstanding_[pipe]);
    return prefix;
  }

  bool scalar_serial_;
  size_t cut_ = 0;
  Ranks outstanding_{};
  std::array<Prefix, kPipeCount> gates_{}, commands_{};
  std::map<std::tuple<PipeType, PipeType, int>, Bank> banks_;
};

}  // namespace pypto::ir::sync::replay

#endif  // SRC_IR_TRANSFORMS_UTILS_PHYSICAL_SYNC_REPLAY_H_
