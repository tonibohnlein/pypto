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

// Canonical direct-cut token transport. This parser proves the relation
// between actual SSA token uses and original payload occurrences; it does not
// regenerate storage demands. Numerical frontiers and allocation are shared.
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/rotating_sync_analysis.h"
#include "src/ir/transforms/utils/scoped_token_families.h"
#include "src/ir/transforms/utils/scoped_token_order.h"
#include "src/ir/transforms/utils/sync_boundary_graph.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/token_sync_lowering.h"

namespace pypto::ir::sync {
namespace {
using Site = std::pair<size_t, size_t>;
StmtPtr Empty(const Span& span) { return std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, span); }
bool EmptyBody(const StmtPtr& body) {
  const auto seq = As<SeqStmts>(body);
  return seq && seq->stmts_.empty();
}
ExprPtr TokenYield(StmtPtr body) {
  while (auto sequence = As<SeqStmts>(body)) {
    if (sequence->stmts_.size() != 1) return nullptr;
    body = sequence->stmts_.front();
  }
  auto yield = As<YieldStmt>(body);
  if (!yield || yield->value_.size() != 1 || !As<SyncTokenType>(yield->value_.front()->GetType()))
    return nullptr;
  return yield->value_.front();
}
PipeType Pipe(const CallPtr& call, const char* key) {
  for (const auto& [name, value] : call->kwargs_)
    if (name == key) return static_cast<PipeType>(AnyCast<int>(value, name));
  CHECK(false) << "Missing logical synchronization pipe";
  return PipeType::ALL;
}
// The erased function is used for shared payload/native import and, only after
// the parser certifies all actual token cuts, for shared direct-cut emission.
class Erase : public IRMutator {
 public:
  explicit Erase(bool allow_nested = false) : allow_nested_(allow_nested) {}
  std::map<const ForStmt*, ForStmtPtr> loops;

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& op) override {
    return RewriteSyncSequence(op, [this](const StmtPtr& child) { return VisitStmt(child); });
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& op) override {
    return As<SyncTokenType>(op->var_->GetType()) ? Empty(op->span_) : op;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& op) override {
    return IsOp(As<Call>(op->expr_), "sync.wait") ? Empty(op->span_) : op;
  }
  StmtPtr VisitStmt_(const IfStmtPtr& op) override {
    for (const auto& result : op->return_vars_) {
      if (!As<SyncTokenType>(result->GetType())) continue;
      CHECK(op->return_vars_.size() == 1 && op->else_body_ && TokenYield(op->then_body_) &&
            TokenYield(*op->else_body_))
          << "Token capture cannot contain payload or scalar effects";
      return Empty(op->span_);  // Pure token selection; the parser must qualify its recurrence.
    }
    auto body = VisitStmt(op->then_body_);
    auto other = op->else_body_ ? VisitStmt(*op->else_body_) : nullptr;
    if (EmptyBody(body) && (!other || EmptyBody(other))) return Empty(op->span_);
    // Non-token branches remain original macros, including both arms.
    return op;
  }
  StmtPtr VisitStmt_(const ForStmtPtr& op) override {
    CHECK(allow_nested_ || !active_) << "LowerSync nested token frames are unavailable";
    CHECK(op->iter_args_.size() == op->return_vars_.size()) << "Token loop result arity mismatch";
    for (const auto& arg : op->iter_args_)
      CHECK(As<SyncTokenType>(arg->GetType())) << "LowerSync non-token loop carries are unavailable";
    for (const auto& var : op->return_vars_)
      CHECK(As<SyncTokenType>(var->GetType())) << "LowerSync non-token loop results are unavailable";
    const bool parent = active_;
    active_ = true;
    auto body = VisitStmt(op->body_);
    active_ = parent;
    auto loop =
        std::make_shared<ForStmt>(op->loop_var_, op->start_, op->stop_, op->step_, std::vector<IterArgPtr>{},
                                  body, std::vector<VarPtr>{}, op->span_, op->kind_, op->attrs_,
                                  op->leading_comments_, op->hardware_loop_policy_);
    loops.emplace(op.get(), loop);
    return loop;
  }
  StmtPtr VisitStmt_(const YieldStmtPtr& op) override {
    for (const auto& value : op->value_)
      CHECK(As<SyncTokenType>(value->GetType())) << "LowerSync non-token yields are unavailable";
    return std::make_shared<YieldStmt>(std::vector<ExprPtr>{}, op->span_, op->leading_comments_);
  }

 private:
  bool active_ = false;
  bool allow_nested_;
};
enum class BindingKind { Completed, Current, Previous, Final, Captured };
struct Binding {
  BindingKind kind = BindingKind::Completed;
  Site site{};
  int64_t lag = 0;
  int64_t visit = 0;
};
enum class Guard { Always, Point, Later };
struct Wait {
  ExprPtr token;
  PipeType destination;
  Guard guard;
  int64_t lag = 0;
  int64_t visit = 0;
};
struct CaptureEquation {
  Binding record;
  const Var* previous;
  int64_t visit;
};
struct Use {
  Wait wait;
  Site consumer;
};
struct Cursor {
  std::optional<size_t> loop;
  std::optional<Site> last;
  std::vector<Wait> pending;
  YieldStmtPtr yield;
  bool returned = false;
};
class Parse {
 public:
  Parse(const FunctionPtr& original, const CompactSyncInput& input,
        const std::map<const ForStmt*, ForStmtPtr>& erased)
      : original_(original), input_(input) {
    CHECK(!input.region_tree) << "LowerSync nested regional token transport is unavailable";
    std::map<const ForStmt*, size_t> regions;
    for (size_t r = 0; r < input.segments.size(); ++r) {
      const auto& segment = input.segments[r];
      CHECK(segment.nested_frames.empty()) << "LowerSync nested token frames are unavailable";
      if (segment.original_loop) regions.emplace(segment.original_loop.get(), r);
      for (size_t s = 0; s < segment.sites.size(); ++s)
        CHECK(sites_.emplace(segment.sites[s].original.get(), Site{r, s}).second)
            << "LowerSync requires distinct original payload anchors";
    }
    for (const auto& [original_loop, stripped] : erased) {
      const auto found = regions.find(stripped.get());
      CHECK(found != regions.end()) << "Missing original token loop frame";
      loops_.emplace(original_loop, found->second);
    }
  }
  std::vector<PhysicalSyncFamily> Run() {
    Cursor cursor;
    Read(original_->body_, cursor);
    CHECK(cursor.pending.empty()) << "Logical wait has no direct consumer cut";
    std::vector<PhysicalSyncFamily> families;
    for (const auto& use : uses_) {
      auto binding = Resolve(use.wait.token);
      if (binding.kind == BindingKind::Completed) continue;
      const auto [sr, ss] = binding.site;
      const auto [tr, ts] = use.consumer;
      const auto& source = input_.segments[sr];
      const auto& target = input_.segments[tr];
      if (!source.count || !target.count) continue;
      int64_t first = 0, target_first = 0, count = 0;
      if (sr == tr) {
        CHECK(binding.kind == BindingKind::Current || binding.kind == BindingKind::Previous)
            << "Token result re-enters its own invocation";
        const bool previous = binding.kind == BindingKind::Previous;
        CHECK(use.wait.guard == (previous ? Guard::Later : Guard::Always) &&
              (!previous || use.wait.lag == binding.lag))
            << "Unsupported token activation: expected exact current/previous visit guard";
        CHECK(previous || ss < ts) << "Logical token does not precede its direct consumer";
        target_first = previous ? binding.lag : 0;
        count = source.count - target_first;
      } else {
        CHECK(sr < tr && ((binding.kind == BindingKind::Final && binding.lag == 1) ||
                          binding.kind == BindingKind::Captured ||
                          (binding.kind == BindingKind::Current && !source.original_loop)))
            << "Unavailable cross-invocation token frame";
        CHECK(use.wait.guard == (target.original_loop ? Guard::Point : Guard::Always))
            << "Crossing token requires an exact target visit";
        first = binding.kind == BindingKind::Captured ? binding.visit : source.count - 1;
        target_first = use.wait.visit;
        if (target_first >= target.count) continue;
        count = 1;
      }
      if (count <= 0) continue;
      families.push_back({sr, ss, tr, ts, first, target_first, count, {}});
    }
    return families;
  }

 private:
  FunctionPtr original_;
  const CompactSyncInput& input_;
  std::map<const Call*, Site> sites_;
  std::map<const ForStmt*, size_t> loops_;
  std::map<const Var*, Binding> bindings_;
  std::map<const Var*, ExprPtr> aliases_;
  std::map<const Var*, std::pair<size_t, ExprPtr>> shifts_;
  std::map<const Var*, CaptureEquation> captures_;
  std::vector<Use> uses_;
  Binding Resolve(const ExprPtr& expression) {
    // Each equation has one successor. Resolve iteratively, then memoize the
    // path in reverse; explicit large histories do not consume the C++ stack.
    std::vector<std::pair<const Var*, std::optional<size_t>>> path;
    std::set<const Var*> seen;
    ExprPtr next = expression;
    Binding binding;
    while (true) {
      const auto var = AsVarLike(next);
      CHECK(var) << "Structured token transport requires named completion facts";
      const auto bound = bindings_.find(var.get());
      if (bound != bindings_.end()) {
        binding = bound->second;
        break;
      }
      CHECK(!captures_.count(var.get())) << "Captured token may only be used through its final loop result";
      CHECK(seen.insert(var.get()).second) << "Cyclic token history does not establish a current record";
      const auto shift = shifts_.find(var.get());
      if (shift != shifts_.end()) {
        path.emplace_back(var.get(), shift->second.first);
        next = shift->second.second;
      } else {
        const auto alias = aliases_.find(var.get());
        CHECK(alias != aliases_.end()) << "Unavailable structured token binding";
        path.emplace_back(var.get(), std::nullopt);
        next = alias->second;
      }
    }
    for (auto step = path.rbegin(); step != path.rend(); ++step) {
      if (step->second) {
        CHECK((binding.kind == BindingKind::Current || binding.kind == BindingKind::Previous) &&
              binding.site.first == *step->second && binding.lag < std::numeric_limits<int64_t>::max())
            << "Token history must end at this invocation's unconditional current record";
        ++binding.lag;
        binding.kind = BindingKind::Previous;
      }
      bindings_.emplace(step->first, binding);
    }
    return binding;
  }
  CallPtr Payload(const StmtPtr& stmt) const {
    if (auto eval = As<EvalStmt>(stmt)) return As<Call>(eval->expr_);
    if (auto branch = As<IfStmt>(stmt)) {
      auto first = branch->then_body_;
      if (auto seq = As<SeqStmts>(first)) {
        if (seq->stmts_.size() != 1) return nullptr;
        first = seq->stmts_.front();
      }
      if (auto eval = As<EvalStmt>(first)) return As<Call>(eval->expr_);
    }
    return nullptr;
  }
  void Waits(const StmtPtr& stmt, Guard guard, Cursor& cursor, int64_t lag = 0, int64_t visit = 0) {
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) Waits(child, guard, cursor, lag, visit);
      return;
    }
    auto eval = As<EvalStmt>(stmt);
    auto call = eval ? As<Call>(eval->expr_) : nullptr;
    CHECK(IsOp(call, "sync.wait")) << "Unsupported conditional token operation";
    cursor.pending.push_back({call->args_.at(0), Pipe(call, "dst_pipe"), guard, lag, visit});
    cursor.last.reset();
  }
  void Read(const StmtPtr& stmt, Cursor& cursor) {
    if (auto seq = As<SeqStmts>(stmt)) {
      for (const auto& child : seq->stmts_) Read(child, cursor);
      return;
    }
    CHECK(!cursor.returned && !cursor.yield) << "Unreachable structured token cut";
    auto call = Payload(stmt);
    const auto payload = sites_.find(call.get());
    if (payload != sites_.end()) {
      const auto site = payload->second;
      CHECK(!cursor.loop || *cursor.loop == site.first) << "Incompatible original token occurrence frame";
      for (const auto& wait : cursor.pending) {
        CHECK(wait.destination == input_.segments[site.first].sites[site.second].pipe)
            << "Logical wait is not at its destination payload cut";
        uses_.push_back({wait, site});
      }
      cursor.pending.clear();
      cursor.last = site;
      return;
    }
    if (auto eval = As<EvalStmt>(stmt)) {
      Waits(eval, Guard::Always, cursor);
      return;
    }
    if (auto branch = As<IfStmt>(stmt)) {
      CHECK(cursor.loop) << "Conditional token transport requires its original loop";
      const auto& segment = input_.segments[*cursor.loop];
      auto binary = As<BinaryExpr>(branch->condition_);
      const auto start = binary ? As<ConstInt>(binary->right_) : nullptr;
      CHECK(binary && start && binary->left_ == segment.original_loop->loop_var_ &&
            (As<Eq>(branch->condition_) || As<Gt>(branch->condition_)))
          << "Unsupported token guard: original first/later coordinate proof is unavailable";
      const __int128 offset = __int128(start->value_) - segment.start;
      CHECK(segment.step > 0 && offset >= 0 && offset % segment.step == 0 &&
            offset / segment.step < std::numeric_limits<int64_t>::max())
          << "Unsupported token guard: normalized coordinate is unproved";
      const int64_t visit = static_cast<int64_t>(offset / segment.step);
      if (!branch->return_vars_.empty()) {
        CHECK(cursor.pending.empty() && branch->return_vars_.size() == 1 && branch->else_body_ &&
              As<Eq>(branch->condition_))
            << "Unavailable token capture control";
        const auto current = TokenYield(branch->then_body_);
        const auto previous = TokenYield(*branch->else_body_);
        CHECK(current && previous && AsVarLike(previous))
            << "Capture must select a current record or its carry";
        auto record = Resolve(current);
        CHECK(record.kind == BindingKind::Current && record.site.first == *cursor.loop)
            << "Capture needs this invocation's unconditional current record";
        captures_.emplace(branch->return_vars_.front().get(),
                          CaptureEquation{record, AsVarLike(previous).get(), visit});
        cursor.last.reset();
        return;
      }
      CHECK(!branch->else_body_) << "Unsupported conditional token transport";
      if (As<Eq>(branch->condition_)) {
        Waits(branch->then_body_, Guard::Point, cursor, 0, visit);
      } else {
        Waits(branch->then_body_, Guard::Later, cursor, visit + 1);
      }
      return;
    }
    if (auto assign = As<AssignStmt>(stmt)) {
      if (As<SyncTokenType>(assign->var_->GetType())) {
        const auto token = As<Call>(assign->value_);
        if (IsOp(token, "sync.record")) {
          CHECK(cursor.last && cursor.pending.empty()) << "Token record is not at a direct payload cut";
          const auto [r, s] = *cursor.last;
          CHECK(Pipe(token, "pipe") == input_.segments[r].sites[s].pipe)
              << "Record pipe differs from its direct payload";
          bindings_.emplace(assign->var_.get(), Binding{BindingKind::Current, {r, s}});
        } else if (IsOp(token, "sync.completed")) {
          bindings_.emplace(assign->var_.get(), Binding{});
        } else {
          CHECK(AsVarLike(assign->value_)) << "Structured token joins require a prefix qualification";
          aliases_.emplace(assign->var_.get(), assign->value_);
        }
      } else {
        CHECK(cursor.pending.empty()) << "Scalar operation separates a token wait from its consumer";
        cursor.last.reset();
      }
      return;
    }
    CHECK(cursor.pending.empty()) << "Token wait crosses an unsupported control cut";
    cursor.last.reset();
    if (auto loop = As<ForStmt>(stmt)) {
      CHECK(!cursor.loop) << "Nested structured token frames are unavailable";
      const auto region = loops_.at(loop.get());
      const auto count = input_.segments[region].count;
      for (const auto& arg : loop->iter_args_)
        CHECK(Resolve(arg->initValue_).kind == BindingKind::Completed)
            << "Structured token carry requires completed entry";
      Cursor body;
      body.loop = region;
      if (count) Read(loop->body_, body);
      CHECK(body.pending.empty()) << "Token wait at loop exit has no direct payload consumer";
      CHECK(!count || (loop->iter_args_.empty() && (!body.yield || body.yield->value_.empty())) ||
            (body.yield && body.yield->value_.size() == loop->iter_args_.size()))
          << "Unavailable unconditional token yield";
      if (count) {
        for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
          const auto yielded = AsVarLike(body.yield->value_[i]);
          const auto captured = yielded ? captures_.find(yielded.get()) : captures_.end();
          if (captured == captures_.end()) {
            shifts_.emplace(loop->iter_args_[i].get(), std::make_pair(region, body.yield->value_[i]));
          } else {
            const auto& capture = captured->second;
            CHECK(capture.previous == loop->iter_args_[i].get() && capture.record.site.first == region)
                << "Capture must preserve its own original invocation carry";
            auto binding = capture.record;
            binding.kind = BindingKind::Captured;
            binding.visit = capture.visit;
            if (capture.visit >= count) binding = Binding{};
            bindings_.emplace(loop->return_vars_[i].get(), binding);
          }
        }
      }
      for (size_t i = 0; i < loop->iter_args_.size(); ++i) {
        if (bindings_.count(loop->return_vars_[i].get())) continue;
        auto binding = count ? Resolve(loop->iter_args_[i]) : Binding{};
        if (count && binding.lag <= count)
          binding.kind = BindingKind::Final;
        else
          binding = Binding{};
        bindings_.emplace(loop->return_vars_[i].get(), binding);
      }
      return;
    }
    if (auto yield = As<YieldStmt>(stmt)) {
      CHECK(cursor.loop) << "Token yield outside original loop";
      cursor.yield = yield;
      return;
    }
    if (As<ReturnStmt>(stmt)) {
      CHECK(!cursor.loop) << "Return inside token loop is unavailable";
      cursor.returned = true;
      return;
    }
    CHECK(false) << "Unsupported structured token statement";
  }
};
// Retained numerical order derives only from actual token uses. Storage access
// and cover fields are intentionally not constructed or exposed as a result.
class Order {
 public:
  Order(const CompactSyncInput& input, const std::vector<PhysicalSyncFamily>& families) {
    segments_.resize(input.segments.size());
    for (size_t r = 0; r < segments_.size(); ++r) {
      auto& segment = segments_[r];
      segment.count = input.segments[r].count;
      segment.core_frontiers = true;
      for (const auto& site : input.segments[r].sites) segment.pipes.push_back(site.pipe);
    }
    for (const auto& f : families)
      if (f.source_region == f.target_region)
        segments_[f.source_region].generators.push_back(
            {f.source_site, f.target_site, f.target_first - f.source_first});
    for (auto& segment : segments_) BuildPeriodicCoreFrontiers(segment);
    if (segments_.size() > 1) {
      std::vector<Occurrence> ports;
      for (const auto& family : families) {
        if (family.source_region == family.target_region) continue;
        ports.emplace_back(family.source_region, family.source_first, family.source_site);
        ports.emplace_back(family.target_region, family.target_first, family.target_site);
      }
      graph_ = std::make_unique<SyncBoundaryGraph>(segments_, ports);
    }
    for (const auto& f : families)
      if (f.source_region != f.target_region)
        graph_->Add({{f.source_region, f.source_first, f.source_site}, true},
                    {{f.target_region, f.target_first, f.target_site}, false});
  }
  DirectSyncOrder View() const {
    return {[&](const Occurrence& a, const Occurrence& b) {
              const auto [ar, av, as] = a;
              const auto [br, bv, bs] = b;
              if (ar == br) return segments_[ar].Precedes(av, as, true, bv, bs, false);
              return ar < br && graph_->Precedes(segments_, a, true, b, false);
            },
            [](const Occurrence& a, const Occurrence& b) { return a < b; },
            [&](size_t r, size_t a, size_t b) { return segments_[r].Threshold(a, true, b, false); },
            {},
            {},
            false};
  }

 private:
  std::vector<CompactSegmentResult> segments_;
  std::unique_ptr<SyncBoundaryGraph> graph_;
};
}  // namespace
StrippedLogicalSync StripStructuredLogicalSync(const FunctionPtr& function, bool allow_nested) {
  Erase erase(allow_nested);
  auto stripped = erase.VisitFunction(function);
  return {std::move(stripped), std::move(erase.loops)};
}
LoweredLogicalSync LowerStructuredLogicalSync(const FunctionPtr& function, const std::string& target,
                                              const std::optional<std::vector<int>>& eligible,
                                              const SyncParameterBounds& parameters,
                                              std::optional<std::pair<int64_t, int64_t>> blocks,
                                              const SyncSchedulePolicy& policy) {
  auto stripped = StripStructuredLogicalSync(function, true);
  if (blocks) {
    const auto partition = PartitionSyncInvocation(stripped.function, parameters, *blocks);
    if (partition.intervals.size() > 1) {
      std::vector<std::unique_ptr<ScopedTokenOrder>> owners;
      std::vector<PhysicalSyncRealization> plans;
      plans.reserve(partition.intervals.size());
      std::vector<AssignedSyncCase> cases;
      LoweredLogicalSync result;
      for (const auto& domain : partition.intervals) {
        auto input = ImportSyncPayloads(stripped.function, target, parameters, domain);
        auto families = ParseScopedTokenFamilies(function, input, stripped.loops);
        owners.push_back(std::make_unique<ScopedTokenOrder>(std::move(input), std::move(families)));
        const auto& order = *owners.back();
        plans.push_back(AllocateDirectSync(order.Input(), order.Families(), order.View(), target, eligible));
        cases.push_back({&order.Input(), &plans.back()});
      }
      result.function = EmitAssignedSyncCases(stripped.function, parameters, *blocks, cases, policy);
      for (const auto& plan : plans) {
        result.set_sites += plan.set_sites;
        result.wait_sites += plan.wait_sites;
      }
      result.physical_cases = std::move(plans);
      return result;
    }
  }
  auto input = ImportSyncPayloads(stripped.function, target, parameters, blocks);
  if (input.region_tree) {
    auto families = ParseScopedTokenFamilies(function, input, stripped.loops);
    ScopedTokenOrder order(std::move(input), std::move(families));
    // The callbacks borrow this owner until shared allocation/emission finishes.
    // An unavailable proof refuses; it never substitutes a new demand analysis.
    auto result = RealizeDirectSync(order.Input(), order.Families(), order.View(), target, eligible, policy);
    return {result.function, result.set_sites, result.wait_sites, {std::move(result)}};
  }
  auto families = Parse(function, input, stripped.loops).Run();
  Order order(input, families);
  auto result = RealizeDirectSync(input, std::move(families), order.View(), target, eligible, policy);
  return {result.function, result.set_sites, result.wait_sites, {std::move(result)}};
}
}  // namespace pypto::ir::sync
