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
  ExprPtr capture;
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
    CHECK(analysis.State().input.region_tree) << "Scoped logical transport requires an original region tree";
    static_branches_ = analysis.State().input.static_branches;
    for (const auto& segment : analysis.State().input.segments)
      for (const auto& frame : segment.nested_frames)
        if (frame.count) active_loops_.insert(frame.loop.get());
    const auto families = analysis.ExportScopedDemands();
    CHECK(families) << "Original-frame logical demand export is unavailable";
    for (const auto& family : families->families) {
      CHECK(analysis.ReferenceBefore(family.source, family.target))
          << "Scoped token source must precede its consumer in the original invocation";
      Add(analysis.State().input, family, guard);
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
    if (const auto resolved = static_branches_.find(op.get()); resolved != static_branches_.end()) {
      if (resolved->second) return VisitStmt(op->then_body_);
      if (op->else_body_) return VisitStmt(*op->else_body_);
      return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, op->span_);
    }
    // Qualified initialization/accumulation is one payload macro. Never move
    // a case record into an arm where its SSA fact would become conditional.
    // Only pointers already certified by the access importer identify the
    // payload. Definitions and the original terminal yield stay in the macro.
    auto arm = [this](const StmtPtr& body) -> const Call* {
      const Call* found = nullptr;
      std::function<void(const StmtPtr&)> find = [&](const StmtPtr& statement) {
        if (auto sequence = As<SeqStmts>(statement)) {
          for (const auto& child : sequence->stmts_) find(child);
        } else if (auto eval = As<EvalStmt>(statement)) {
          const auto call = As<Call>(eval->expr_);
          if (cuts_.count(call.get())) {
            CHECK(!found) << "Scoped logical macro has multiple certified payload cuts";
            found = call.get();
          }
        }
      };
      find(body);
      return found;
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
    std::vector<IterArgPtr> arguments = op->iter_args_;
    std::vector<VarPtr> outputs = op->return_vars_;
    for (const auto& transport : transports_[op.get()]) {
      statements.push_back(std::make_shared<AssignStmt>(
          transport.initial,
          Token("sync.completed", As<SyncTokenType>(transport.initial->GetType())->pipe_, op->span_),
          op->span_));
      arguments.push_back(transport.previous);
      outputs.push_back(transport.exported);
    }
    CHECK(op->iter_args_.size() == op->return_vars_.size())
        << "Scoped token insertion requires consistent original loop results";
    statements.push_back(std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, arguments,
                                                   body, outputs, op->span_, op->kind_, op->attrs_,
                                                   op->leading_comments_, op->hardware_loop_policy_));
    return SeqStmts::Flatten(statements, op->span_);
  }
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    CHECK(active_) << "Scoped token yield has no original loop frame";
    std::vector<StmtPtr> statements;
    std::vector<ExprPtr> values = op->value_;
    for (const auto& transport : transports_[active_]) {
      if (transport.capture) {
        const auto& condition = transport.capture;
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
  std::map<const IfStmt*, bool> static_branches_;
  std::map<const Call*, std::shared_ptr<LogicalCut>> cuts_;
  std::map<const ForStmt*, std::vector<Transport>> transports_;

  VarPtr Variable(PipeType pipe) {
    return std::make_shared<Var>(names_.Next(), std::make_shared<SyncTokenType>(pipe), Span::unknown());
  }
  static CallPtr Token(const char* name, PipeType pipe, const Span& span) {
    return OpRegistry::GetInstance().Create(name, {}, {{"pipe", static_cast<int>(pipe)}}, span);
  }
  // Exact ordinal progression, expressed in the original available IV frame.
  // Normalizing at the loop start keeps the remainder operand nonnegative on
  // every execution, even before the first selected source occurrence.
  static ExprPtr Domain(const SyncLoopFrame& frame, int64_t first, int64_t count, int64_t step) {
    const auto last = __int128(first) + (__int128(count) - 1) * step;
    CHECK(first >= 0 && count > 0 && step > 0 && last < frame.count)
        << "Scoped token activity exceeds its original frame";
    Coordinate(frame, frame.count - 1);
    if (count == frame.count && step == 1) return nullptr;
    const auto& span = frame.loop->span_;
    const auto& iv = frame.loop->loop_var_;
    if (count == 1) return std::make_shared<Eq>(iv, Coordinate(frame, first), DataType::BOOL, span);
    ExprPtr guard;
    auto add = [&](ExprPtr value) {
      guard = guard ? std::make_shared<And>(guard, value, DataType::BOOL, span) : value;
    };
    if (first) add(std::make_shared<Gt>(iv, Coordinate(frame, first - 1), DataType::BOOL, span));
    if (last + 1 < frame.count)
      add(std::make_shared<Lt>(iv, Coordinate(frame, static_cast<int64_t>(last + 1)), DataType::BOOL, span));
    if (step > 1) {
      const auto modulus = __int128(step) * frame.step;
      CHECK(modulus <= std::numeric_limits<int64_t>::max()) << "Scoped token residue modulus overflows";
      const auto normalized = std::make_shared<Sub>(iv, Coordinate(frame, 0), DataType::INDEX, span);
      const auto remainder = std::make_shared<FloorMod>(
          normalized, std::make_shared<ConstInt>(static_cast<int64_t>(modulus), DataType::INDEX, span),
          DataType::INDEX, span);
      add(std::make_shared<Eq>(remainder,
                               std::make_shared<ConstInt>((first % step) * frame.step, DataType::INDEX, span),
                               DataType::BOOL, span));
    }
    return guard;
  }
  Transport Carry(const SyncLoopFrame& frame, const ExprPtr& value, const ExprPtr& capture) {
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
        const auto source_last = __int128(a.digits[i]) + (__int128(axis.count) - 1) * axis.source_step;
        const auto target_last = __int128(b.digits[i]) + (__int128(axis.count) - 1) * axis.target_step;
        CHECK(axis.frame == source[i].loop && axis.count > 0 && axis.source_step > 0 &&
              axis.target_step == axis.source_step &&
              __int128(axis.source_stride) == __int128(a.strides[i]) * axis.source_step &&
              __int128(axis.target_stride) == __int128(b.strides[i]) * axis.target_step &&
              source_last < source[i].count && target_last < target[i].count)
            << "Scoped token axis does not match the original occurrence map";
      }
      if (!shift && a.digits[i] != b.digits[i]) {
        const auto delta = b.digits[i] - a.digits[i];
        const auto period = i < family.axes.size() ? family.axes[i].source_step : delta;
        CHECK(delta > 0 && delta <= period)
            << "Scoped token transport requires the latest captured source, not arbitrary history";
        // Axes beneath the carry would need several simultaneous histories.
        CHECK(i + 1 >= family.axes.size())
            << "Scoped token capture cannot collapse a varying inner source coordinate";
        shift = i;
      }
    }
    auto& source_cut = Cut(input.segments.at(sr).sites.at(ss));
    auto& record = source_cut.record;
    if (!record) record = Variable(source_cut.pipe);
    ExprPtr fact = record;
    const size_t keep = shift ? *shift + 1 : common;
    for (size_t i = source.size(); i-- > keep;) {
      // Source suffix coordinates are fixed at this seam. The selected fact
      // survives subsequent iterations without inventing a completion event.
      ExprPtr capture;
      if (a.digits[i] + 1 != source[i].count) capture = Domain(source[i], a.digits[i], 1, 1);
      fact = Carry(source[i], fact, capture).exported;
    }
    if (shift) {
      const auto i = *shift;
      const auto count = i < family.axes.size() ? family.axes[i].count : 1;
      const auto step = i < family.axes.size() ? family.axes[i].source_step : 1;
      fact = Carry(source[i], fact, Domain(source[i], a.digits[i], count, step)).previous;
    }
    ExprPtr guard = case_guard;
    for (size_t i = 0; i < target.size(); ++i) {
      const auto count = i < family.axes.size() ? family.axes[i].count : 1;
      const auto step = i < family.axes.size() ? family.axes[i].target_step : 1;
      const auto condition = Domain(target[i], b.digits[i], count, step);
      if (condition)
        guard = guard ? std::make_shared<And>(guard, condition, DataType::BOOL, Span::unknown()) : condition;
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
  CHECK(analysis.State().input.region_tree) << "Scoped logical transport requires an original region tree";
  ScopedRealizer realizer(analysis.State().input.owner);
  realizer.Append(analysis);
  realizer.result.function = realizer.VisitFunction(analysis.State().input.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
LogicalSyncRealization RealizeLogicalInvocation(const SyncInvocationAnalysis& analysis) {
  CHECK(analysis.owner && !analysis.cases.empty()) << "Missing logical invocation owner";
  ScopedRealizer realizer(analysis.owner);
  const auto block =
      analysis.coordinate ? analysis.coordinate->Expression() : DominatingSyncBlock(analysis.owner);
  auto next = analysis.block_bounds.first;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics && item.mathematics->State().input.owner == analysis.owner &&
          item.mathematics->State().input.parameter_bounds == analysis.parameter_bounds &&
          item.mathematics->State().input.invocation_coordinate == analysis.coordinate &&
          item.mathematics->State().input.block_bounds == std::make_pair(item.block_first, item.block_last) &&
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
  // completed. Captures use proved original-coordinate progressions;
  // dynamic exits use only the original available start expression.
  // Thus unused case state is total even outside its case. Only waits add order.
  realizer.result.function = realizer.VisitFunction(analysis.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
}  // namespace pypto::ir::sync
