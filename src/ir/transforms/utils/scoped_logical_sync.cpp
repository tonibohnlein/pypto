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
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {
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
    // The importer proves a uniform ordinal domain and original-width range
    // for every caller. Reconstruct the original IV, retaining its SSA start.
    const auto offset = __int128(ordinal) * frame.step;
    CHECK(ordinal >= 0 && ordinal < frame.count && frame.step > 0 &&
          offset <= std::numeric_limits<int64_t>::max())
        << "Scoped token dynamic coordinate exceeds its proved frame";
    if (!offset) return frame.start_expression;
    return std::make_shared<Add>(
        frame.start_expression,
        std::make_shared<ConstInt>(static_cast<int64_t>(offset), DataType::INDEX, frame.loop->span_),
        DataType::INDEX, frame.loop->span_);
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

struct LogicalCut {
  PipeType pipe;
  VarPtr record;
  std::vector<Incoming> incoming;
};
class ScopedRealizer : public IRMutator {
 public:
  explicit ScopedRealizer(const FunctionPtr& owner) : names_(owner) {
    result.name_scan_visits = names_.visits;
    result.name_scan_nodes = names_.Nodes();
  }
  void Append(const CompactSyncResult& analysis, const ExprPtr& guard = nullptr) {
    CHECK(analysis.input.region_tree) << "Scoped logical transport requires an original region tree";
    for (const auto& segment : analysis.input.segments)
      for (const auto& frame : segment.nested_frames)
        if (frame.count) active_loops_.insert(frame.loop.get());
    const auto families = analysis.ExportScopedDemands();
    CHECK(families) << "Original-frame logical demand export is unavailable";
    for (const auto& family : families->families) {
      CHECK(analysis.ReferenceBefore(family.source, family.target))
          << "Scoped token source must precede its consumer in the original invocation";
      Add(analysis.input, family, guard);
    }
  }
  LogicalSyncRealization result;

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return RewriteSyncSequence(op, [this](const StmtPtr& child) { return VisitStmt(child); });
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override { return op; }
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    const auto call = As<Call>(op->expr_);
    auto found = cuts_.find(call.get());
    return found == cuts_.end() ? op : Wrap(op, *found->second);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    // Qualified initialization/accumulation is one payload macro. Never move
    // a case record into an arm where its SSA fact would become conditional.
    auto arm = [](StmtPtr body) -> const Call* {
      while (auto sequence = As<SeqStmts>(body)) {
        if (sequence->stmts_.size() != 1) return nullptr;
        body = sequence->stmts_.front();
      }
      const auto eval = As<EvalStmt>(body);
      return eval ? As<Call>(eval->expr_).get() : nullptr;
    };
    auto found = cuts_.find(arm(op->then_body_));
    if (found == cuts_.end()) return op;
    CHECK(op->else_body_) << "Scoped logical macro requires original initialization alternatives";
    const auto other = cuts_.find(arm(*op->else_body_));
    CHECK(other != cuts_.end() && other->second == found->second)
        << "Scoped logical initialization alternatives have incompatible cuts";
    return Wrap(op, *found->second);
  }
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    // Inventories are keyed by actual loop identity, not a representative leaf.
    if (!active_loops_.count(op.get())) return op;
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
  std::set<const ForStmt*> active_loops_;
  std::map<const Call*, std::shared_ptr<LogicalCut>> cuts_;
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
  LogicalCut& Cut(const SyncSite& site) {
    auto& cut = cuts_[site.original.get()];
    if (!cut) cut = std::make_shared<LogicalCut>(LogicalCut{site.pipe, {}, {}});
    CHECK(cut->pipe == site.pipe) << "Invocation cases disagree on original payload pipe";
    if (site.continuation) {
      auto& continuation = cuts_[site.continuation.get()];
      CHECK(!continuation || continuation == cut) << "Invocation cases disagree on original macro cut";
      continuation = cut;
    }
    return *cut;
  }
  void Add(const CompactSyncInput& input, const ScopedDemandFamily& family, const ExprPtr& case_guard) {
    const auto [sr, sv, ss] = family.source;
    const auto [tr, tv, ts] = family.target;
    const auto& source = input.segments.at(sr).nested_frames;
    const auto& target = input.segments.at(tr).nested_frames;
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
    auto& source_cut = Cut(input.segments.at(sr).sites.at(ss));
    auto& record = source_cut.record;
    if (!record) record = Variable(source_cut.pipe);
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
    ExprPtr guard = case_guard;
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
    Cut(input.segments.at(tr).sites.at(ts)).incoming.push_back({fact, guard});
  }
  StmtPtr Wrap(const StmtPtr& payload, const LogicalCut& cut) {
    std::vector<StmtPtr> statements;
    for (const auto& incoming : cut.incoming) {
      const auto wait = OpRegistry::GetInstance().Create(
          "sync.wait", {incoming.token}, {{"dst_pipe", static_cast<int>(cut.pipe)}}, payload->span_);
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
    if (cut.record) {
      statements.push_back(std::make_shared<AssignStmt>(
          cut.record, Token("sync.record", cut.pipe, payload->span_), payload->span_));
      ++result.record_sites;
    }
    return SeqStmts::Flatten(statements, payload->span_);
  }
};
}  // namespace
LogicalSyncRealization RealizeScopedLogicalSync(const CompactSyncResult& analysis) {
  CHECK(analysis.input.region_tree) << "Scoped logical transport requires an original region tree";
  ScopedRealizer realizer(analysis.input.owner);
  realizer.Append(analysis);
  realizer.result.function = realizer.VisitFunction(analysis.input.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
LogicalSyncRealization RealizeLogicalInvocation(const SyncInvocationAnalysis& analysis) {
  CHECK(analysis.owner && !analysis.cases.empty()) << "Missing logical invocation owner";
  ScopedRealizer realizer(analysis.owner);
  const auto block = DominatingSyncBlock(analysis.owner);
  auto next = analysis.block_bounds.first;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics && item.mathematics->input.owner == analysis.owner &&
          item.mathematics->input.parameter_bounds == analysis.parameter_bounds &&
          item.mathematics->input.block_bounds == std::make_pair(item.block_first, item.block_last) &&
          item.block_first == next && item.block_last >= next &&
          item.block_last < std::numeric_limits<int64_t>::max())
        << "Incomplete logical invocation cases";
    next = item.block_last + 1;
    auto guard = SyncInvocationGuard(block, analysis.block_bounds, {item.block_first, item.block_last},
                                     analysis.owner->span_);
    realizer.Append(*item.mathematics, guard);
  }
  CHECK(next == analysis.block_bounds.second + 1) << "Incomplete logical invocation domain";
  // All records are after original payload macros, and every carry starts
  // completed. Case-independent captures use constant original coordinates;
  // dynamic singleton exits use only the original available start expression.
  // Thus unused case state is total even outside its case. Only waits add order.
  realizer.result.function = realizer.VisitFunction(analysis.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
}  // namespace pypto::ir::sync
