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

#include "src/ir/transforms/utils/peer_local_work_loop.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "src/ir/transforms/utils/closed_prefix_sync.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"

namespace pypto::ir::sync {
namespace {

constexpr int64_t kMovingSlot = -1, kAllSlots = -2;
struct Effect {
  size_t storage;
  int64_t slot, begin, end;
  PipeType pipe;
  bool write, following;
};

// All access coordinates of a runtime work item have the same incoming ring
// rotation. After any number of items the difference in origin is a multiple
// of gcd(depth, pops/item). Fixed residues still distinguish disjoint slots,
// including the phase-neutral case. A varying residue in a finite child loop
// uses the whole orbit so crossing analysis never enumerates loop trips.
void CollectEffects(const CompactSyncInput& input, const PeerLocalWorkLoop& loop, PeerWorkRange range,
                    bool following, std::vector<Effect>& effects) {
  for (size_t r = range.first; r < range.second; ++r) {
    const auto& segment = input.segments[r];
    if (!segment.count) {
      continue;
    }
    for (const auto& site : segment.sites) {
      for (const auto& access : site.accesses) {
        const auto [modulus, begin, end] = ProjectPeerWorkAccess(input, loop, access);
        const auto slot =
            segment.count > 1 && access.slot_step % modulus ? kMovingSlot : access.slot_first % modulus;
        effects.push_back({access.storage, slot, begin, end, site.pipe, access.write, following});
      }
    }
  }
}

using EffectKey = std::pair<size_t, int64_t>;
using PipeEnds = std::array<int64_t, kPipeCount>;
using PrefixEnds = std::array<std::array<PipeEnds, 2>, 2>;

void AddOverlaps(const PrefixEnds& ends, const Effect& effect, PeerWorkPairs& result) {
  const auto& other = ends[!effect.following];
  for (size_t p = 0; p < kPipeCount; ++p) {
    if (other[true][p] > effect.begin || (effect.write && other[false][p] > effect.begin)) {
      const auto pipe = static_cast<PipeType>(p);
      if (effect.pipe != PipeType::S || pipe != PipeType::S) {
        result.emplace(std::min(pipe, effect.pipe), std::max(pipe, effect.pipe));
      }
    }
  }
}

void Remember(PrefixEnds& ends, const Effect& effect) {
  auto& end = ends[effect.following][effect.write][static_cast<size_t>(effect.pipe)];
  end = std::max(end, effect.end);
}

StmtPtr MakeFlag(const char* op, SyncDirection pipes, int id, const std::string& core, const Span& span) {
  const auto call = OpRegistry::GetInstance().Create(op, {},
                                                     {{"src_pipe", static_cast<int>(pipes.first)},
                                                      {"dst_pipe", static_cast<int>(pipes.second)},
                                                      {"event_id", id},
                                                      {"core_type", core}},
                                                     span);
  return std::make_shared<EvalStmt>(call, span);
}

void AppendSeam(std::vector<StmtPtr>& stmts, const std::vector<SyncDirection>& commands,
                const PeerWorkSeamIds& ids, FunctionType type, const Span& span) {
  const std::string core = type == FunctionType::AIV ? "aiv" : "aic";
  for (const auto& direction : commands) {
    const auto [source, target] = direction;
    if (source != target) {
      stmts.push_back(MakeFlag("system.set_flag", direction, ids.at(direction), core, span));
      stmts.push_back(MakeFlag("system.wait_flag", direction, ids.at(direction), core, span));
    } else {
      const auto op = RequireLocalSyncBarrier(source, type);
      INTERNAL_CHECK(op) << "Peer work-item seam must not request an explicit scalar barrier";
      const auto call = OpRegistry::GetInstance().Create(op, {}, {{"core_type", core}}, span);
      stmts.push_back(std::make_shared<EvalStmt>(call, span));
    }
  }
}

class WorkLoopSeams : public SyncStatementMutator {
 public:
  WorkLoopSeams(const std::vector<PeerWorkPlacement>& placements, const PeerWorkSeamIds& ids,
                FunctionType type)
      : ids_(ids), type_(type) {
    for (const auto& placement : placements) {
      INTERNAL_CHECK(placements_.emplace(placement.loop->original->loop_var_.get(), &placement).second)
          << "Duplicate original runtime work-loop cut";
    }
  }

  StmtPtr Run(const StmtPtr& body) {
    auto result = VisitStmt(body);
    INTERNAL_CHECK(placements_.empty()) << "Original runtime work-loop cut was lost during local planning";
    return result;
  }

 private:
  std::unordered_map<const Var*, const PeerWorkPlacement*> placements_;
  const PeerWorkSeamIds& ids_;
  const FunctionType type_;

  StmtPtr CompleteItem(const StmtPtr& original, const std::vector<SyncDirection>& exit,
                       const Span& span) const {
    auto flattened = original;
    if (const auto seq = As<SeqStmts>(original)) {
      // A terminal yield can sit under arbitrarily nested transparent
      // sequences. Do not enter child loops: they own their own yield.
      flattened = RewriteSyncSequence(seq, [](const StmtPtr& stmt) { return stmt; });
    }
    auto body = transform_utils::FlattenToStmts(flattened);
    StmtPtr yield;
    if (!body.empty() && IsA<YieldStmt>(body.back())) {
      yield = body.back();
      body.pop_back();
    }
    AppendSeam(body, exit, ids_, type_, span);
    if (yield) {
      body.push_back(std::move(yield));
    }
    return SeqStmts::Flatten(std::move(body), original->span_);
  }

  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    const auto found = placements_.find(loop->loop_var_.get());
    if (found == placements_.end()) {
      return SyncStatementMutator::VisitStmt_(loop);
    }
    const auto& placement = *found->second;
    placements_.erase(found);
    StmtPtr updated = loop;
    if (!placement.exit.empty()) {
      auto with_exit = MutableCopy(loop);
      with_exit->body_ = CompleteItem(loop->body_, placement.exit, loop->span_);
      updated = std::move(with_exit);
    }
    std::vector<StmtPtr> stmts;
    AppendSeam(stmts, placement.entry, ids_, type_, loop->span_);
    stmts.push_back(std::move(updated));
    return SeqStmts::Flatten(std::move(stmts), loop->span_);
  }
};

}  // namespace

PeerWorkAccess ProjectPeerWorkAccess(const CompactSyncInput& input, const PeerLocalWorkLoop& loop,
                                     const SyncAccess& access) {
  INTERNAL_CHECK(access.row == 0 && access.rows == 1 && !access.row_step && !access.column_step &&
                 access.column >= 0 && access.columns > 0 && access.storage < input.storage.size());
  const auto& storage = input.storage[access.storage];
  INTERNAL_CHECK(access.column <= storage.columns && access.columns <= storage.columns - access.column);
  const auto pop = loop.ring_advance.find(access.storage);
  const auto modulus = std::gcd(storage.slot_count, pop == loop.ring_advance.end() ? 0 : pop->second);
  INTERNAL_CHECK(modulus > 0 && access.slot_first >= 0 && access.slot_step >= 0);
  return {modulus, access.column, access.column + access.columns};
}

void AddPeerWorkConflicts(const CompactSyncInput& input, const PeerLocalWorkLoop& loop, PeerWorkRange earlier,
                          PeerWorkRange later, PeerWorkPairs& pairs) {
  std::vector<Effect> effects;
  CollectEffects(input, loop, earlier, false, effects);
  CollectEffects(input, loop, later, true, effects);
  std::sort(effects.begin(), effects.end(), [](const auto& a, const auto& b) {
    return std::tie(a.storage, a.begin) < std::tie(b.storage, b.begin);
  });
  // Half-open byte intervals. The maximum end of each earlier-starting group
  // suffices to test all opposite-phase read/write overlaps at this cut.
  std::map<EffectKey, PrefixEnds> seen;
  for (const auto& effect : effects) {
    const EffectKey all{effect.storage, kAllSlots}, exact{effect.storage, effect.slot};
    if (effect.slot == kMovingSlot) {
      AddOverlaps(seen[all], effect, pairs);
    } else {
      AddOverlaps(seen[exact], effect, pairs);
      AddOverlaps(seen[{effect.storage, kMovingSlot}], effect, pairs);
    }
    Remember(seen[exact], effect);
    Remember(seen[all], effect);
  }
}

std::vector<SyncDirection> ClosePeerWorkPairs(const PeerWorkPairs& pairs) {
  std::vector<SyncDirection> commands;
  std::set<PipeType> completed;
  // A sets, B consumes and acknowledges, then A consumes the reply. Both
  // flags can then be republished after another visit to this boundary.
  for (const auto& [source, target] : pairs) {
    if (source != target) {
      commands.emplace_back(source, target);
      commands.emplace_back(target, source);
      completed.insert(source);
    }
  }
  for (const auto& [source, target] : pairs) {
    if (source == target && !completed.count(source)) {
      commands.emplace_back(source, target);
    }
  }
  return commands;
}

FunctionPtr AddPeerWorkSeams(const FunctionPtr& function, const std::vector<PeerWorkPlacement>& placements,
                             const PeerWorkSeamIds& ids) {
  auto result = MutableCopy(function);
  result->body_ = WorkLoopSeams(placements, ids, function->func_type_).Run(function->body_);
  return result;
}

}  // namespace pypto::ir::sync
