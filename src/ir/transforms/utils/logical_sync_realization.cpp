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
  VarPtr current, initial, exported;
  IterArgPtr previous;
  bool carry = false;
};
struct Incoming {
  Site source;
  bool previous = false, crossing = false;
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

// One indexed IR traversal, O((N+F) log(N+F)). The owner has already proved
// invariant sites and zero/one-displacement families; no trip count is unfolded.
class Realizer : public SyncPlacement {
 public:
  explicit Realizer(const CompactSyncResult& analysis) : SyncPlacement(analysis) {
    CHECK(analysis.input.owner) << "Logical realization requires the original occurrence owner";
    NameCollector collector;
    collector.VisitFunction(analysis.input.owner);
    result.name_scan_visits = collector.visits;
    result.name_scan_nodes = collector.seen.size();
    for (const auto* var : collector.GetAllVarRefs()) names_.insert(var->name_hint_);
    for (size_t r = 0; r < analysis.input.segments.size(); ++r) {
      const auto& segment = analysis.input.segments[r];
      for (const auto& f : analysis.segments[r].demands) {
        CHECK(f.distance == 0 || f.distance == 1) << "Unavailable logical token displacement";
        if (segment.count <= f.distance) continue;
        Site source{r, f.source}, target{r, f.target};
        sources_[source].carry |= f.distance != 0;
        incoming_[target].push_back({source, f.distance != 0, false});
      }
    }
    for (const auto& [a, b] : analysis.crossing_demands) {
      auto [ar, av, as] = a;
      auto [br, bv, bs] = b;
      CHECK(ar < br && av == analysis.input.segments[ar].count - 1 && bv == 0)
          << "Crossing token export requires final-source/first-target coordinates";
      Site source{ar, as};
      sources_[source].carry |= analysis.input.segments[ar].original_loop != nullptr;
      incoming_[{br, bs}].push_back({source, false, true});
    }
    for (auto& [site, source] : sources_) {
      const auto pipe = Payload(site).pipe;
      const auto type = std::make_shared<SyncTokenType>(pipe);
      source.current = Variable(type);
      if (source.carry) {
        source.initial = Variable(type);
        source.previous = std::make_shared<IterArg>(Name(), type, source.initial, Span::unknown());
        source.exported = Variable(type);
        carried_[site.first].push_back(site);
        ++result.carried_tokens;
      }
    }
  }
  LogicalSyncRealization result;

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    const size_t region = loops_.at(op.get());
    if (analysis_.input.segments[region].count == 0) return op;
    std::vector<StmtPtr> statements;
    std::vector<IterArgPtr> arguments;
    std::vector<VarPtr> outputs;
    for (auto site : carried_[region]) {
      const auto& source = sources_.at(site);
      statements.push_back(std::make_shared<AssignStmt>(
          source.initial, Token("sync.completed", Payload(site).pipe, op->span_), op->span_));
      arguments.push_back(source.previous);
      outputs.push_back(source.exported);
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
    for (auto site : carried_[*active_loop_]) values.push_back(sources_.at(site).current);
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
    const auto& segment = analysis_.input.segments[site.first];
    for (const auto& incoming : incoming_[site]) {
      const auto& source = sources_.at(incoming.source);
      ExprPtr token = source.current;
      if (incoming.previous) token = source.previous;
      if (incoming.crossing && source.carry) token = source.exported;
      auto wait = OpRegistry::GetInstance().Create(
          "sync.wait", {token}, {{"dst_pipe", static_cast<int>(Payload(site).pipe)}}, payload->span_);
      StmtPtr statement = std::make_shared<EvalStmt>(wait, payload->span_);
      ++result.wait_sites;
      if (segment.original_loop && (incoming.previous || incoming.crossing)) {
        auto start = std::make_shared<ConstInt>(segment.start, DataType::INDEX, payload->span_);
        ExprPtr condition;
        if (incoming.previous)
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
  Realizer realizer(analysis);
  realizer.result.function = realizer.VisitFunction(analysis.input.owner);
  CheckLogicalSync(realizer.result.function);
  return realizer.result;
}
}  // namespace pypto::ir::sync
