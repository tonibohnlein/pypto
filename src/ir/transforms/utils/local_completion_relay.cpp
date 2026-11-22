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

#include "src/ir/transforms/utils/local_completion_relay.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_sync_contract.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/utils/transform_utils.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_transform_common.h"

namespace pypto::ir::sync {
namespace {
constexpr size_t kPipes = static_cast<size_t>(PipeType::ALL);

bool IsAsyncPipe(PipeType pipe) {
  return pipe >= PipeType::MTE1 && pipe < PipeType::ALL && pipe != PipeType::S;
}

// Track prefix ranks within a single unconditional window. A WAIT follows
// older payload issue on its queue without draining their completion. Only
// the completion rank also implies issue of that and earlier FIFO payloads.
using Ranks = std::array<size_t, kPipes>;
struct Prefix {
  Ranks issue{};
  Ranks payload{};
  Ranks entry_gates{};
  Ranks entry_commands{};
  Ranks wait{};

  void Include(const Prefix& other) {
    for (size_t p = 0; p < kPipes; ++p) {
      issue[p] = std::max(issue[p], other.issue[p]);
      payload[p] = std::max(payload[p], other.payload[p]);
      entry_gates[p] = std::max(entry_gates[p], other.entry_gates[p]);
      entry_commands[p] = std::max(entry_commands[p], other.entry_commands[p]);
      wait[p] = std::max(wait[p], other.wait[p]);
    }
  }

  bool Preserves(const Prefix& baseline) const {
    // Preserve all original payload dependencies without adding any. Preserve
    // surviving consumption ancestry too, so the allocator's proof for every
    // retained event still holds after intermediate services disappear.
    if (payload != baseline.payload || entry_gates != baseline.entry_gates ||
        entry_commands != baseline.entry_commands) {
      return false;
    }
    for (size_t p = 0; p < kPipes; ++p) {
      if (std::max(payload[p], issue[p]) != std::max(baseline.payload[p], baseline.issue[p]) ||
          wait[p] < baseline.wait[p]) {
        return false;
      }
    }
    return true;
  }
};

struct QueuePrefixes {
  QueuePrefixes() {
    // Arbitrary incoming payload and WAIT frontiers survive across the window.
    // Giving each queue its own symbols proves the rewrite for every entry,
    // including either branch arm and every visit to a loop.
    for (size_t p = 0; p < kPipes; ++p) {
      outstanding[p] = 1;
      gate[p].entry_gates[p] = 1;
      gate[p].wait[p] = 1;
      commands[p] = gate[p];
      commands[p].entry_commands[p] = 1;
    }
  }

  Prefix Capture(PipeType source) const {
    auto prefix = commands[source];
    prefix.Include(gate[source]);
    prefix.payload[source] = std::max(prefix.payload[source], outstanding[source]);
    return prefix;
  }

  void Apply(PipeType source, PipeType target, size_t retained_wait = 0) {
    commands[source] = Capture(source);
    gate[target].Include(commands[source]);
    gate[target].issue[target] = std::max(gate[target].issue[target], outstanding[target]);
    if (retained_wait != 0) gate[target].wait[target] = retained_wait;
    commands[target].Include(gate[target]);
  }

  bool Preserves(const QueuePrefixes& baseline) const {
    for (size_t p = 0; p < kPipes; ++p) {
      // Opaque continuations can observe command completion without draining
      // outstanding payload. Preserve both interfaces on every queue.
      if (!gate[p].Preserves(baseline.gate[p]) || !commands[p].Preserves(baseline.commands[p])) return false;
    }
    return true;
  }

  std::array<Prefix, kPipes> gate;
  std::array<Prefix, kPipes> commands;
  Ranks outstanding{};
};

struct Flag {
  SyncDirection direction;
  int id;
  bool publish;
};

StmtPtr MakeFlag(bool publish, SyncDirection direction, int id, const std::string& core, const Span& span) {
  const auto call = OpRegistry::GetInstance().Create(publish ? "system.set_flag" : "system.wait_flag", {},
                                                     {{"src_pipe", static_cast<int>(direction.first)},
                                                      {"dst_pipe", static_cast<int>(direction.second)},
                                                      {"event_id", id},
                                                      {"core_type", core}},
                                                     span);
  return std::make_shared<EvalStmt>(call, span);
}

std::optional<Flag> ReadFlag(const StmtPtr& statement) {
  const auto eval = As<EvalStmt>(statement);
  const auto call = eval ? As<Call>(eval->expr_) : nullptr;
  if (!call || !call->args_.empty()) return std::nullopt;
  const bool publish = IsOp(call, "system.set_flag");
  if (!publish && !IsOp(call, "system.wait_flag")) return std::nullopt;
  return Flag{{RequireSyncPipeKwarg(call, "src_pipe"), RequireSyncPipeKwarg(call, "dst_pipe")},
              call->GetKwarg<int>("event_id"),
              publish};
}

std::optional<Flag> GeneratedFlag(const StmtPtr& stmt, const ImportedLocalCuts& cuts,
                                  const SyncNativeReservations& native) {
  if (cuts.count(transform_utils::GetCallFromStmt(stmt).get())) return std::nullopt;
  const auto flag = ReadFlag(stmt);
  if (!flag) return std::nullopt;
  const auto [source, destination] = flag->direction;
  if (source == destination || !IsAsyncPipe(source) || !IsAsyncPipe(destination)) return std::nullopt;
  const auto claimed = native.find(flag->direction);
  if (claimed != native.end() && claimed->second.count(flag->id)) return std::nullopt;
  return flag;
}

std::optional<SyncDirection> GeneratedPair(const std::vector<StmtPtr>& body, size_t at, size_t end,
                                           const ImportedLocalCuts& cuts,
                                           const SyncNativeReservations& native) {
  if (end - at < 2) return std::nullopt;
  const auto set = GeneratedFlag(body[at], cuts, native);
  if (!set || !set->publish) return std::nullopt;
  const auto wait = GeneratedFlag(body[at + 1], cuts, native);
  return wait && !wait->publish && wait->id == set->id && wait->direction == set->direction
             ? std::optional<SyncDirection>(set->direction)
             : std::nullopt;
}

class AllocatedEvents final : public IRVisitor {
 public:
  explicit AllocatedEvents(const SyncNativeReservations& reservations) : used(reservations) {}

  void VisitStmt(const StmtPtr& statement) override {
    if (const auto flag = ReadFlag(statement)) used[flag->direction].insert(flag->id);
    IRVisitor::VisitStmt(statement);
  }

  SyncNativeReservations used;

 protected:
  void VisitExpr(const ExprPtr&) override {}
};

struct Step {
  enum class Kind { Fence, Metadata, Payload, Pair };
  Kind kind = Kind::Fence;
  SyncDirection direction{};
};

class SelectRelays final : public SyncStatementMutator {
 public:
  SelectRelays(const FunctionPtr& owner, const ImportedLocalCuts& cuts, const std::string& target,
               const SyncNativeReservations& native)
      : owner_(owner),
        cuts_(cuts),
        target_(target),
        native_(native),
        core_(owner->func_type_ == FunctionType::AIC ? "aic" : "aiv") {
    AllocatedEvents allocated(native_);
    allocated.VisitStmt(owner_->body_);
    allocated_ = std::move(allocated.used);
  }

 protected:
  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    ++loop_depth_;
    const auto result = SyncStatementMutator::VisitStmt_(loop);
    --loop_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const WhileStmtPtr& loop) override {
    ++loop_depth_;
    const auto result = SyncStatementMutator::VisitStmt_(loop);
    --loop_depth_;
    return result;
  }

  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto body = transform_utils::FlattenToStmts(sequence);
    const auto steps = Classify(body);
    std::vector<StmtPtr> output;
    output.reserve(body.size());
    QueuePrefixes queues;
    size_t protected_end = 0;
    bool changed = false;
    for (size_t i = 0; i < body.size();) {
      const auto& step = steps[i];
      if (step.kind == Step::Kind::Pair) {
        const auto destination = step.direction.second;
        const size_t end = BatchEnd(steps, i, destination);
        if (i >= protected_end && TryBatch(body, steps, i, end, destination, queues, protected_end, output)) {
          changed = true;
        } else {
          for (size_t at = i; at < end; at += 2) {
            queues.Apply(steps[at].direction.first, destination, at + 2);
          }
          output.insert(output.end(), body.begin() + i, body.begin() + end);
        }
        i = end;
        continue;
      }

      const auto result = VisitStmt(body[i]);
      changed = changed || result != body[i];
      output.push_back(result);
      if (step.kind == Step::Kind::Payload) {
        queues.outstanding[step.direction.first] = i + 2;
      } else if (step.kind == Step::Kind::Fence) {
        queues = QueuePrefixes{};
      }
      ++i;
    }
    return changed ? std::make_shared<SeqStmts>(std::move(output), sequence->span_) : sequence;
  }

 private:
  std::vector<Step> Classify(const std::vector<StmtPtr>& body) const {
    std::vector<Step> steps(body.size());
    for (size_t i = 0; i < body.size(); ++i) {
      const auto call = transform_utils::GetCallFromStmt(body[i]);
      const auto imported = cuts_.find(call.get());
      if (imported != cuts_.end()) {
        const auto& cut = imported->second;
        if (cut.statement != body[i].get() || cut.HasOpaqueProtocol()) continue;
        if (cut.pipe && IsAsyncPipe(*cut.pipe)) {
          steps[i] = {Step::Kind::Payload, {*cut.pipe, *cut.pipe}};
        } else if (!cut.pipe && backend::QualifyBufferSyncNonPayload(call, target_)) {
          steps[i].kind = Step::Kind::Metadata;
        }
        continue;
      }
      if (const auto pair = GeneratedPair(body, i, body.size(), cuts_, native_)) {
        steps[i] = {Step::Kind::Pair, *pair};
        ++i;
      }
    }
    return steps;
  }

  static size_t BatchEnd(const std::vector<Step>& steps, size_t start, PipeType destination) {
    size_t end = start + 2;
    while (end < steps.size() && steps[end].kind == Step::Kind::Pair &&
           steps[end].direction.second == destination) {
      end += 2;
    }
    return end;
  }

  // A fresh static lane may be republished by a later visit. Require a
  // forward-in-issue-order corridor that transfers the new consumption to
  // its publishing queue before leaving this unconditional scope. Keep that
  // whole corridor unchanged so later choices cannot invalidate the proof.
  std::optional<size_t> Rearm(const std::vector<Step>& steps, size_t begin, PipeType carrier,
                              PipeType destination) const {
    if (loop_depth_ == 0) return begin;
    constexpr size_t kMaxRearmStatements = 64;
    std::array<bool, kPipes> reached{};
    reached[destination] = true;
    for (size_t at = begin; at < steps.size() && at - begin < kMaxRearmStatements;) {
      const auto& step = steps[at];
      if (step.kind == Step::Kind::Fence) break;
      if (step.kind == Step::Kind::Pair) {
        const auto [source, target] = step.direction;
        reached[target] = reached[target] || reached[source];
        at += 2;
        if (reached[carrier]) return at;
      } else {
        ++at;
      }
    }
    return std::nullopt;
  }

  bool TryBatch(const std::vector<StmtPtr>& body, const std::vector<Step>& steps, size_t begin, size_t end,
                PipeType destination, QueuePrefixes& queues, size_t& protected_end,
                std::vector<StmtPtr>& output) {
    auto baseline = queues;
    // All SET commands read a queue other than the shared destination. No
    // payload, opaque operation or read from that destination is interleaved.
    // The original capture set is frozen before choosing any replacement.
    std::array<std::optional<size_t>, kPipes> direct;
    for (size_t at = begin; at < end; at += 2) {
      const auto source = steps[at].direction.first;
      baseline.Apply(source, destination);
      direct[source] = at;
    }

    // A locally redundant batch needs no event. Its disappearing WAITs have
    // no remaining users as event consumptions. The ancestry of every retained
    // WAIT that reached these cuts remains available through the same gate.
    if (queues.Preserves(baseline)) return true;
    if (end - begin == 2) return false;

    for (size_t p = 0; p < kPipes; ++p) {
      if (!direct[p]) continue;
      auto replacement = queues;
      replacement.Apply(static_cast<PipeType>(p), destination);
      if (!replacement.Preserves(baseline)) continue;
      const size_t at = *direct[p];
      queues.Apply(static_cast<PipeType>(p), destination, at + 2);
      output.push_back(body[at]);
      output.push_back(body[at + 1]);
      return true;
    }

    for (size_t p = 0; p < kPipes; ++p) {
      const auto carrier = static_cast<PipeType>(p);
      if (!IsAsyncPipe(carrier) || carrier == destination || direct[p]) continue;
      auto replacement = queues;
      replacement.Apply(carrier, destination);
      if (!replacement.Preserves(baseline)) continue;
      const auto rearm = Rearm(steps, end, carrier, destination);
      if (!rearm) continue;
      const SyncDirection direction{carrier, destination};
      const auto* catalog = pypto::sync::FindResourceCatalog(target_);
      const auto core =
          owner_->func_type_ == FunctionType::AIC ? pypto::sync::CoreScope::AIC : pypto::sync::CoreScope::AIV;
      if (!catalog ||
          !catalog->FindPool({pypto::sync::ResourceKind::PipelineEvent, core, carrier, destination}))
        continue;
      const auto eligible = SyncEligibleIds(owner_, target_, direction, std::nullopt, allocated_);
      if (eligible.empty()) continue;
      const int id = eligible.front();
      allocated_[direction].insert(id);
      queues = replacement;
      const auto& span = body[begin]->span_;
      output.push_back(MakeFlag(true, direction, id, core_, span));
      output.push_back(MakeFlag(false, direction, id, core_, span));
      protected_end = *rearm;
      return true;
    }
    // Keep a smaller direct subset when a single eligible relay could not
    // provide the same payload order. There is at most one distinct capture
    // per source in this batch, so even repeated pairs incur bounded work per
    // source. Keep the selected original pairs in their allocated order.
    auto retained = direct;
    for (size_t p = 0; p < kPipes; ++p) {
      if (!retained[p]) continue;
      const auto original = std::exchange(retained[p], std::nullopt);
      auto replacement = queues;
      for (size_t source = 0; source < kPipes; ++source) {
        if (retained[source]) replacement.Apply(static_cast<PipeType>(source), destination);
      }
      if (!replacement.Preserves(baseline)) retained[p] = original;
    }
    const auto count =
        std::count_if(retained.begin(), retained.end(), [](const auto& cut) { return cut.has_value(); });
    if (static_cast<size_t>(count) == (end - begin) / 2) return false;
    for (size_t at = begin; at < end; at += 2) {
      const auto source = steps[at].direction.first;
      if (retained[source] != at) continue;
      queues.Apply(source, destination, at + 2);
      output.push_back(body[at]);
      output.push_back(body[at + 1]);
    }
    return true;
  }

  const FunctionPtr& owner_;
  const ImportedLocalCuts& cuts_;
  const std::string& target_;
  const SyncNativeReservations& native_;
  const std::string core_;
  SyncNativeReservations allocated_;
  size_t loop_depth_ = 0;
};

bool Conflicts(const ImportedLocalCut& producer, const ImportedLocalCut& consumer) {
  for (const auto& from : producer.effects) {
    for (const auto& to : consumer.effects) {
      if (from.space != to.space || (!from.write && !to.write)) {
        continue;
      }
      // DDR roots can alias. For local memory a missing enclosure is a
      // wildcard and a present enclosure covers every reachable placement.
      if (from.space == MemorySpace::DDR || !from.enclosure || !to.enclosure ||
          (from.enclosure->begin < to.enclosure->end && to.enclosure->begin < from.enclosure->end)) {
        return true;
      }
    }
  }
  return false;
}

// Split an already allocated capture across two consumers on the same
// queue. The original publication is kept, so later consumers still see
// the exact same source prefix. An existing reverse handoff at the head
// of each disjoint window orders consumption of the new event (from any
// preceding window or visit) before its next publication.
class SplitCaptures final : public SyncStatementMutator {
 public:
  SplitCaptures(const FunctionPtr& owner, const ImportedLocalCuts& cuts, const std::string& target,
                const SyncNativeReservations& native)
      : owner_(owner),
        cuts_(cuts),
        native_(native),
        target_(target),
        core_(owner->func_type_ == FunctionType::AIC ? "aic" : "aiv"),
        occupied_(native) {
    occupied_.VisitStmt(owner->body_);
  }

 protected:
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    const auto& body = sequence->stmts_;
    std::vector<StmtPtr> output;
    output.reserve(body.size());
    bool changed = false;
    for (size_t at = 0; at < body.size();) {
      if (const auto split = FindSplit(body, at)) {
        if (const auto id = Allocate(split->direction)) {
          AppendSplit(body, at, *split, *id, output);
          at = split->right + 1;
          changed = true;
          continue;
        }
      }
      auto result = VisitStmt(body[at]);
      changed = changed || result != body[at];
      output.push_back(std::move(result));
      ++at;
    }
    return changed ? std::make_shared<SeqStmts>(std::move(output), sequence->span_) : sequence;
  }

 private:
  struct Split {
    SyncDirection direction;
    size_t record;
    size_t wait;
    size_t right;
  };

  const ImportedLocalCut* Payload(const StmtPtr& stmt, PipeType pipe) const {
    const auto found = cuts_.find(transform_utils::GetCallFromStmt(stmt).get());
    if (found == cuts_.end()) {
      return nullptr;
    }
    const auto& cut = found->second;
    return cut.statement == stmt.get() && cut.pipe == pipe && !cut.HasOpaqueProtocol() && !cut.effects.empty()
               ? &cut
               : nullptr;
  }

  bool Metadata(const StmtPtr& stmt) const {
    if (const auto call = transform_utils::GetCallFromStmt(stmt)) {
      return backend::QualifyBufferSyncNonPayload(call, target_).has_value();
    }
    return static_cast<bool>(As<AssignStmt>(stmt));
  }

  std::optional<Split> FindSplit(const std::vector<StmtPtr>& body, size_t begin) const {
    constexpr size_t kMaxWindowStatements = 64;
    const auto end = begin + std::min(kMaxWindowStatements, body.size() - begin);
    const auto reverse = GeneratedPair(body, begin, end, cuts_, native_);
    if (!reverse) {
      return std::nullopt;
    }
    const auto [destination, source] = *reverse;
    const SyncDirection direction{source, destination};
    // No commands acquire another queue's completion between the two
    // publications. Producer payloads in this gap are the only possible
    // difference in the completion facts received by the first consumer.
    std::vector<size_t> producers;
    std::optional<int> published;
    for (size_t at = begin + 2; at < end; ++at) {
      if (const auto flag = GeneratedFlag(body[at], cuts_, native_); flag && flag->direction == direction) {
        if (flag->publish && !published && !producers.empty()) {
          published = flag->id;
          continue;
        }
        if (!flag->publish && flag->id == published) {
          return MatchConsumers(body, at, end, direction, producers);
        }
        break;
      }
      if (Payload(body[at], source)) {
        if (!published) {
          producers.push_back(at);
        }
      } else if (!Metadata(body[at])) {
        break;
      }
    }
    return std::nullopt;
  }

  std::optional<Split> MatchConsumers(const std::vector<StmtPtr>& body, size_t wait, size_t end,
                                      SyncDirection direction, const std::vector<size_t>& producers) const {
    const auto [source, destination] = direction;
    const ImportedLocalCut* first = nullptr;
    for (size_t at = wait + 1; at < end; ++at) {
      if (Metadata(body[at])) {
        continue;
      }
      if (const auto consumer = Payload(body[at], destination)) {
        if (first) {
          const auto record = IndependentSuffix(body, source, *first, *consumer, producers);
          return record ? std::optional<Split>{{direction, *record, wait, at}} : std::nullopt;
        }
        first = consumer;
        continue;
      }
      // Independent inputs to the first consumer stay at their existing
      // positions. A read of the destination queue, or any other observable
      // command between the consumers, could observe the delayed completion.
      const auto pair = GeneratedPair(body, at, end, cuts_, native_);
      if (first || !pair || pair->second != destination || pair->first == source) {
        break;
      }
      ++at;
    }
    return std::nullopt;
  }

  std::optional<size_t> IndependentSuffix(const std::vector<StmtPtr>& body, PipeType source,
                                          const ImportedLocalCut& first, const ImportedLocalCut& second,
                                          const std::vector<size_t>& producers) const {
    std::optional<size_t> record;
    bool helps = false;
    for (auto it = producers.rbegin(); it != producers.rend(); ++it) {
      const auto* producer = Payload(body[*it], source);
      if (!producer || Conflicts(*producer, first)) {
        break;
      }
      helps = helps || Conflicts(*producer, second);
      record = *it;
    }
    return helps ? record : std::nullopt;
  }

  std::optional<int> Allocate(SyncDirection direction) {
    const auto [found, inserted] = split_events_.try_emplace(direction);
    if (inserted) {
      const auto ids = SyncEligibleIds(owner_, target_, direction, std::nullopt, occupied_.used);
      if (!ids.empty()) {
        found->second = ids.front();
      }
    }
    return found->second;
  }

  void AppendSplit(const std::vector<StmtPtr>& body, size_t begin, const Split& split, int id,
                   std::vector<StmtPtr>& output) const {
    for (size_t at = begin; at <= split.right; ++at) {
      if (at == split.record) {
        output.push_back(MakeFlag(true, split.direction, id, core_, body[at]->span_));
      }
      if (at == split.wait) {
        output.push_back(MakeFlag(false, split.direction, id, core_, body[at]->span_));
        continue;
      }
      if (at == split.right) {
        output.push_back(body[split.wait]);
      }
      output.push_back(body[at]);
    }
  }

  const FunctionPtr& owner_;
  const ImportedLocalCuts& cuts_;
  const SyncNativeReservations& native_;
  const std::string& target_;
  const std::string core_;
  AllocatedEvents occupied_;
  std::map<SyncDirection, std::optional<int>> split_events_;
};
}  // namespace

FunctionPtr RefineLocalCompletionPlan(const FunctionPtr& physical, const ImportedLocalCuts& cuts,
                                      const std::string& target,
                                      const SyncNativeReservations& native_reservations) {
  auto selector = SelectRelays(physical, cuts, target, native_reservations);
  const auto body = selector.VisitStmt(physical->body_);
  FunctionPtr selected = physical;
  if (body != physical->body_) {
    auto result = MutableCopy(physical);
    result->body_ = body;
    selected = std::move(result);
  }
  return SplitCaptures(selected, cuts, target, native_reservations).VisitFunction(selected);
}
}  // namespace pypto::ir::sync
