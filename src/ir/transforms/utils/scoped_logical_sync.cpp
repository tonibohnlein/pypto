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

// Original-frame logical transport of retained scoped demand families. State
// scales with supplied families and traversed frame depth, never trip counts.
// Each family keeps its matching identity; record sharing only shares a fact.
#include "src/ir/transforms/utils/scoped_logical_sync.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/sync_placement.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {
using Site = SyncSiteKey;
struct Coordinates {
  std::vector<int64_t> digits, strides;
};
Coordinates Decode(const std::vector<SyncLoopFrame>& frames, int64_t ordinal) {
  Coordinates out;
  out.digits.resize(frames.size());
  out.strides.resize(frames.size());
  int64_t stride = 1;
  for (size_t i = frames.size(); i-- > 0;) {
    CHECK(frames[i].count > 0 && ordinal >= 0) << "Scoped token endpoint has no active occurrence";
    out.digits[i] = ordinal % frames[i].count;
    ordinal /= frames[i].count;
    out.strides[i] = stride;
    const auto product = __int128(stride) * frames[i].count;
    CHECK(product <= std::numeric_limits<int64_t>::max()) << "Scoped token coordinate product overflows";
    stride = static_cast<int64_t>(product);
  }
  CHECK(!ordinal) << "Scoped token endpoint exceeds its original frame";
  return out;
}
ExprPtr Coordinate(const SyncLoopFrame& frame, int64_t ordinal) {
  if (frame.start_expression && !As<ConstInt>(frame.start_expression)) {
    // Import has proved this original invocation executes exactly once for
    // every admitted caller. Its numerical start is only a placeholder.
    CHECK(frame.count == 1 && ordinal == 0) << "Scoped token dynamic frame is not a proved singleton";
    return frame.start_expression;
  }
  const auto value = __int128(frame.start) + __int128(ordinal) * frame.step;
  CHECK(frame.start >= 0 && frame.step > 0 && value >= 0 && value <= std::numeric_limits<int64_t>::max())
      << "Scoped token original coordinate overflows";
  return std::make_shared<ConstInt>(static_cast<int64_t>(value), DataType::INDEX, frame.loop->span_);
}
struct Transport {
  SyncLoopFrame frame;
  ExprPtr value;
  VarPtr initial, selected, exported;
  IterArgPtr previous;
  std::optional<int64_t> capture;
};
struct Incoming {
  ExprPtr token, condition;
};

class ScopedRealizer : public SyncPlacement {
 public:
  explicit ScopedRealizer(const CompactSyncResult& analysis)
      : SyncPlacement(analysis.input), names_(analysis.input.owner) {
    result.name_scan_visits = names_.visits;
    result.name_scan_nodes = names_.Nodes();
    std::function<void(const SyncRegion&)> collect = [&](const SyncRegion& region) {
      if (region.repetition && !region.repetition->count) absent_.insert(region.repetition->loop.get());
      for (const auto& child : region.children) collect(child);
    };
    collect(*analysis.input.region_tree);
    for (const auto& segment : input_.segments)
      for (const auto& frame : segment.nested_frames)
        if (!frame.count) absent_.insert(frame.loop.get());
    const auto families = analysis.ExportScopedDemands();
    CHECK(families) << "Original-frame logical demand export is unavailable";
    for (const auto& family : families->families) {
      CHECK(analysis.ReferenceBefore(family.source, family.target))
          << "Scoped token source must precede its consumer in the original invocation";
      Add(family);
    }
  }
  LogicalSyncRealization result;

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Inventories are keyed by actual loop identity, not a representative leaf.
    if (absent_.count(op.get())) return op;
    const auto previous = active_;
    active_ = op.get();
    auto body = VisitStmt(op->body_);
    auto seq = As<SeqStmts>(body);
    const bool has_yield =
        As<YieldStmt>(body) || (seq && !seq->stmts_.empty() && As<YieldStmt>(seq->stmts_.back()));
    if (!has_yield)
      body = SeqStmts::Flatten(
          {body, VisitStmt(std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, op->span_))}, op->span_);
    active_ = previous;
    std::vector<StmtPtr> statements;
    std::vector<IterArgPtr> arguments;
    std::vector<VarPtr> outputs;
    for (const auto& transport : transports_[op.get()]) {
      statements.push_back(std::make_shared<AssignStmt>(
          transport.initial,
          Token("sync.completed", As<SyncTokenType>(transport.initial->GetType())->pipe_, op->span_),
          op->span_));
      arguments.push_back(transport.previous);
      outputs.push_back(transport.exported);
    }
    CHECK(op->iter_args_.empty() && op->return_vars_.empty())
        << "Scoped token insertion requires coordinate-only original loops";
    statements.push_back(std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, arguments,
                                                   body, outputs, op->span_, op->kind_, op->attrs_,
                                                   op->leading_comments_, op->hardware_loop_policy_));
    return SeqStmts::Flatten(statements, op->span_);
  }
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    CHECK(active_ && op->value_.empty()) << "Scoped token insertion requires an empty original yield";
    std::vector<StmtPtr> statements;
    std::vector<ExprPtr> values;
    for (const auto& transport : transports_[active_]) {
      if (transport.capture) {
        auto condition =
            std::make_shared<Eq>(transport.frame.loop->loop_var_,
                                 Coordinate(transport.frame, *transport.capture), DataType::BOOL, op->span_);
        auto selected = std::make_shared<YieldStmt>(std::vector<ExprPtr>{transport.value}, op->span_);
        auto previous = std::make_shared<YieldStmt>(std::vector<ExprPtr>{transport.previous}, op->span_);
        statements.push_back(std::make_shared<IfStmt>(condition, selected, previous,
                                                      std::vector<VarPtr>{transport.selected}, op->span_));
        values.push_back(transport.selected);
      } else {
        values.push_back(transport.value);
      }
    }
    statements.push_back(std::make_shared<YieldStmt>(values, op->span_, op->leading_comments_));
    return SeqStmts::Flatten(statements, op->span_);
  }

 private:
  SyncTokenNames names_;
  const ForStmt* active_ = nullptr;
  std::set<const ForStmt*> absent_;
  std::map<Site, VarPtr> records_;
  std::map<Site, std::vector<Incoming>> incoming_;
  std::map<const ForStmt*, std::vector<Transport>> transports_;

  VarPtr Variable(PipeType pipe) {
    return std::make_shared<Var>(names_.Next(), std::make_shared<SyncTokenType>(pipe), Span::unknown());
  }
  static CallPtr Token(const char* name, PipeType pipe, const Span& span) {
    return OpRegistry::GetInstance().Create(name, {}, {{"pipe", static_cast<int>(pipe)}}, span);
  }
  Transport Carry(const SyncLoopFrame& frame, const ExprPtr& value, std::optional<int64_t> capture) {
    Coordinate(frame, frame.count - 1);  // Qualify the complete constant frame, not one sample.
    const auto pipe = As<SyncTokenType>(value->GetType())->pipe_;
    Transport out{frame, value, Variable(pipe), Variable(pipe), Variable(pipe), {}, capture};
    out.previous = std::make_shared<IterArg>(names_.Next(), value->GetType(), out.initial, Span::unknown());
    transports_[frame.loop.get()].push_back(out);
    ++result.carried_tokens;
    return out;
  }
  void Add(const ScopedDemandFamily& family) {
    const auto [sr, sv, ss] = family.source;
    const auto [tr, tv, ts] = family.target;
    const auto& source = input_.segments.at(sr).nested_frames;
    const auto& target = input_.segments.at(tr).nested_frames;
    const auto a = Decode(source, sv), b = Decode(target, tv);
    size_t common = 0;
    while (common < source.size() && common < target.size() && source[common].loop == target[common].loop)
      ++common;
    CHECK(family.axes.size() <= common) << "Scoped token axes must share an original enclosing prefix";
    std::optional<size_t> shift;
    for (size_t i = 0; i < common; ++i) {
      if (i < family.axes.size()) {
        const auto& axis = family.axes[i];
        CHECK(axis.frame == source[i].loop && axis.count > 0 && axis.source_stride == a.strides[i] &&
              axis.target_stride == b.strides[i] && axis.count <= source[i].count - a.digits[i] &&
              axis.count <= target[i].count - b.digits[i])
            << "Scoped token axis does not match the original occurrence map";
        if (a.digits[i] != b.digits[i]) {
          CHECK(!shift && i + 1 == family.axes.size() && !a.digits[i] && b.digits[i] == 1 &&
                axis.count == source[i].count - 1)
              << "Scoped logical tokens support only adjacent common-frame seams";
          shift = i;
        }
      } else {
        // A qualified outer seam exits the entire suffix before carrying the
        // fact; its inner source/target counters may therefore reset. Without
        // that seam, differing fixed digits would need an in-frame history.
        CHECK(shift || a.digits[i] == b.digits[i])
            << "Scoped logical tokens cannot transport differing fixed common-frame occurrences";
      }
    }
    auto& record = records_[{sr, ss}];
    if (!record) record = Variable(Payload({sr, ss}).pipe);
    ExprPtr fact = record;
    const size_t keep = shift ? *shift + 1 : common;
    for (size_t i = source.size(); i-- > keep;) {
      // Last-visit export needs only the ordinary simultaneous yield. Other
      // selected visits retain a pure token capture, scoped to this invocation.
      std::optional<int64_t> capture;
      if (a.digits[i] + 1 != source[i].count) capture = a.digits[i];
      fact = Carry(source[i], fact, capture).exported;
    }
    if (shift) fact = Carry(source[*shift], fact, {}).previous;
    ExprPtr guard;
    auto add_guard = [&](ExprPtr condition) {
      guard = guard ? std::make_shared<And>(guard, condition, DataType::BOOL, Span::unknown()) : condition;
    };
    for (size_t i = 0; i < target.size(); ++i) {
      const auto low = b.digits[i];
      const auto count = i < family.axes.size() ? family.axes[i].count : 1;
      const auto& frame = target[i];
      Coordinate(frame, frame.count - 1);
      if (count == frame.count) continue;
      if (count == 1) {
        add_guard(std::make_shared<Eq>(frame.loop->loop_var_, Coordinate(frame, low), DataType::BOOL,
                                       frame.loop->span_));
      } else {
        if (low)
          add_guard(std::make_shared<Gt>(frame.loop->loop_var_, Coordinate(frame, low - 1), DataType::BOOL,
                                         frame.loop->span_));
        if (low + count < frame.count)
          add_guard(std::make_shared<Lt>(frame.loop->loop_var_, Coordinate(frame, low + count),
                                         DataType::BOOL, frame.loop->span_));
      }
    }
    incoming_[{tr, ts}].push_back({fact, guard});
  }
  StmtPtr Wrap(const StmtPtr& payload, Site site) override {
    std::vector<StmtPtr> statements;
    for (const auto& incoming : incoming_[site]) {
      const auto wait = OpRegistry::GetInstance().Create("sync.wait", {incoming.token},
                                                         {{"dst_pipe", static_cast<int>(Payload(site).pipe)}},
                                                         payload->span_);
      StmtPtr statement = std::make_shared<EvalStmt>(wait, payload->span_);
      if (incoming.condition) {
        statement = std::make_shared<IfStmt>(incoming.condition, statement, std::nullopt,
                                             std::vector<VarPtr>{}, payload->span_);
        ++result.guarded_wait_sites;
      }
      statements.push_back(statement);
      ++result.wait_sites;
    }
    statements.push_back(payload);
    const auto record = records_.find(site);
    if (record != records_.end()) {
      statements.push_back(std::make_shared<AssignStmt>(
          record->second, Token("sync.record", Payload(site).pipe, payload->span_), payload->span_));
      ++result.record_sites;
    }
    return SeqStmts::Flatten(statements, payload->span_);
  }
};
}  // namespace
LogicalSyncRealization RealizeScopedLogicalSync(const CompactSyncResult& analysis) {
  CHECK(analysis.input.region_tree) << "Scoped logical transport requires an original region tree";
  ScopedRealizer realizer(analysis);
  realizer.result.function = realizer.VisitFunction(analysis.input.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
}  // namespace pypto::ir::sync
