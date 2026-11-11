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

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/ir/transforms/utils/var_collectors.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/sync_placement.h"

namespace pypto::ir::sync {
namespace {
using Site = std::pair<size_t, size_t>;  // Original segment and static site.
struct Source {
  VarPtr current, initial;
  std::vector<VarPtr> exported;
  std::vector<IterArgPtr> previous;
  int64_t history = 0;
};
struct Incoming {
  Site source;
  int64_t distance = 0;
  bool crossing = false;
};

// Name reservation is context-independent. Reuse definition/use collection,
// but inspect a shared operand DAG only once instead of expanding its paths.
class NameCollector : public var_collectors::VarDefUseCollector {
 public:
  void VisitExpr(const ExprPtr& expression) override {
    ++visits;
    if (expression && seen.insert(expression.get()).second) IRVisitor::VisitExpr(expression);
  }
  size_t visits = 0;
  std::set<const Expr*> seen;
};

// One indexed IR traversal, O((N+F+H) log(N+F+H)), including H history
// carry/result/yield nodes. The owner has already proved
// invariant sites and supplied displacement families; no trip count is unfolded.
// Scalar RFC carries cost O(sum maximum live lag per source). This is not a
// compact-in-binary-period token container; mathematical reduction stays compact.
class Realizer : public SyncPlacement {
 public:
  explicit Realizer(const CompactSyncResult& analysis) : SyncPlacement(analysis.input) {
    CHECK(analysis.input.owner) << "Logical realization requires the original occurrence owner";
    NameCollector collector;
    collector.VisitFunction(analysis.input.owner);
    result.name_scan_visits = collector.visits;
    result.name_scan_nodes = collector.seen.size();
    for (const auto* var : collector.GetAllVarRefs()) names_.insert(var->name_hint_);
    for (size_t r = 0; r < analysis.input.segments.size(); ++r) {
      const auto& segment = analysis.input.segments[r];
      for (const auto& f : analysis.segments[r].demands) {
        CHECK(f.distance >= 0) << "Invalid logical token displacement";
        if (segment.count <= f.distance) continue;
        Site source{r, f.source}, target{r, f.target};
        sources_[source].history = std::max(sources_[source].history, f.distance);
        incoming_[target].push_back({source, f.distance, false});
      }
    }
    const auto crossings = analysis.ExportCrossingDemands();
    CHECK(crossings) << "Logical crossing token scope is unavailable";
    for (const auto& [a, b] : *crossings) {
      auto [ar, av, as] = a;
      auto [br, bv, bs] = b;
      CHECK(ar < br && av == analysis.input.segments[ar].count - 1 && bv == 0)
          << "Crossing token export requires final-source/first-target coordinates";
      Site source{ar, as};
      auto& retained = sources_[source];
      if (analysis.input.segments[ar].original_loop)
        retained.history = std::max<int64_t>(retained.history, 1);
      incoming_[{br, bs}].push_back({source, 0, true});
    }
    for (auto& [site, source] : sources_) {
      const auto pipe = Payload(site).pipe;
      const auto type = std::make_shared<SyncTokenType>(pipe);
      source.current = Variable(type);
      if (source.history) {
        source.initial = Variable(type);
        for (int64_t lag = 0; lag < source.history; ++lag) {
          source.previous.push_back(std::make_shared<IterArg>(Name(), type, source.initial, Span::unknown()));
          source.exported.push_back(Variable(type));
        }
        carried_[site.first].push_back(site);
        result.carried_tokens += source.previous.size();
      }
    }
  }
  LogicalSyncRealization result;

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const size_t region = loops_.at(op.get());
    if (input_.segments[region].count == 0) return op;
    std::vector<StmtPtr> statements;
    std::vector<IterArgPtr> arguments;
    std::vector<VarPtr> outputs;
    for (auto site : carried_[region]) {
      const auto& source = sources_.at(site);
      statements.push_back(std::make_shared<AssignStmt>(
          source.initial, Token("sync.completed", Payload(site).pipe, op->span_), op->span_));
      arguments.insert(arguments.end(), source.previous.begin(), source.previous.end());
      outputs.insert(outputs.end(), source.exported.begin(), source.exported.end());
    }
    active_loop_ = region;
    auto body = VisitStmt(op->body_);
    auto seq = As<SeqStmts>(body);
    const bool has_yield =
        As<YieldStmt>(body) || (seq && !seq->stmts_.empty() && As<YieldStmt>(seq->stmts_.back()));
    if (!has_yield) {
      auto empty = std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, op->span_);
      body = SeqStmts::Flatten({body, VisitStmt(empty)}, op->span_);
    }
    active_loop_.reset();
    // Importer admits coordinate-only loops; carry only retained completion
    // facts, and preserve all original scheduling attributes and coordinates.
    statements.push_back(std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, arguments,
                                                   body, outputs, op->span_, op->kind_, op->attrs_,
                                                   op->leading_comments_, op->hardware_loop_policy_));
    return std::make_shared<SeqStmts>(statements, op->span_);
  }
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    INTERNAL_CHECK(active_loop_ && op->value_.empty()) << "Unexpected original yield";
    std::vector<ExprPtr> values;
    for (auto site : carried_[*active_loop_]) {
      const auto& source = sources_.at(site);
      values.push_back(source.current);
      // Yield is simultaneous: older entries see the previous visit, never
      // the current record shifted sequentially through all positions.
      for (size_t lag = 1; lag < source.previous.size(); ++lag) values.push_back(source.previous[lag - 1]);
    }
    return std::make_shared<YieldStmt>(values, op->span_, op->leading_comments_);
  }

 private:
  std::map<Site, Source> sources_;
  std::map<Site, std::vector<Incoming>> incoming_;
  std::map<size_t, std::vector<Site>> carried_;
  std::optional<size_t> active_loop_;
  std::set<std::string> names_;
  size_t next_name_ = 0;
  std::string Name() {
    // Monotone counter avoids repeatedly searching all earlier suffixes.
    std::string name;
    do {
      name = "sync_completion_" + std::to_string(next_name_++);
    } while (!names_.insert(name).second);
    return name;
  }
  VarPtr Variable(const TypePtr& type) { return std::make_shared<Var>(Name(), type, Span::unknown()); }
  static CallPtr Token(const char* name, PipeType pipe, const Span& span) {
    return OpRegistry::GetInstance().Create(name, {}, {{"pipe", static_cast<int>(pipe)}}, span);
  }
  StmtPtr Wrap(const StmtPtr& payload, Site site) override {
    std::vector<StmtPtr> statements;
    const auto& segment = input_.segments[site.first];
    for (const auto& incoming : incoming_[site]) {
      const auto& source = sources_.at(incoming.source);
      ExprPtr token = source.current;
      if (incoming.distance) token = source.previous.at(incoming.distance - 1);
      if (incoming.crossing && source.history) token = source.exported.front();
      auto wait = OpRegistry::GetInstance().Create(
          "sync.wait", {token}, {{"dst_pipe", static_cast<int>(Payload(site).pipe)}}, payload->span_);
      StmtPtr statement = std::make_shared<EvalStmt>(wait, payload->span_);
      ++result.wait_sites;
      if (segment.original_loop && (incoming.distance || incoming.crossing)) {
        const __int128 cut = __int128(segment.start) +
                             (incoming.distance ? __int128(incoming.distance - 1) * segment.step : 0);
        CHECK(cut >= 0 && cut <= std::numeric_limits<int64_t>::max())
            << "Logical history activation coordinate overflows";
        auto start = std::make_shared<ConstInt>(static_cast<int64_t>(cut), DataType::INDEX, payload->span_);
        ExprPtr condition;
        if (incoming.distance)
          condition =
              std::make_shared<Gt>(segment.original_loop->loop_var_, start, DataType::BOOL, payload->span_);
        else
          condition =
              std::make_shared<Eq>(segment.original_loop->loop_var_, start, DataType::BOOL, payload->span_);
        statement = std::make_shared<IfStmt>(condition, statement, std::nullopt, std::vector<VarPtr>{},
                                             payload->span_);
        ++result.guarded_wait_sites;
      }
      statements.push_back(statement);
    }
    statements.push_back(payload);
    auto source = sources_.find(site);
    if (source != sources_.end()) {
      statements.push_back(std::make_shared<AssignStmt>(
          source->second.current, Token("sync.record", Payload(site).pipe, payload->span_), payload->span_));
      ++result.record_sites;
    }
    return std::make_shared<SeqStmts>(statements, payload->span_);
  }
};
}  // namespace
LogicalSyncRealization RealizeCompactSync(const CompactSyncResult& analysis) {
  CHECK(!analysis.input.region_tree) << "Nested/phase logical token transport is unavailable";
  for (const auto& segment : analysis.segments)
    CHECK(!segment.rotating || analysis.segments.size() == 1)
        << "Rotating crossing token selectors are unavailable";
  for (const auto& segment : analysis.input.segments)
    CHECK(segment.nested_frames.empty()) << "Nested repetition requires token transport across every "
                                            "enclosing frame; physical export is available";
  Realizer realizer(analysis);
  realizer.result.function = realizer.VisitFunction(analysis.input.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
}  // namespace pypto::ir::sync
