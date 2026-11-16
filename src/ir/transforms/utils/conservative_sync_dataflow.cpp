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

#include "src/ir/transforms/utils/conservative_sync_dataflow.h"

#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/native_sync_placement.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"

namespace pypto::ir::sync {
namespace {
// ParentStmtAnalysis owns the shared statement index. Expressions cannot carry
// admitted statements; skipping them avoids retraversing IterArg initializer
// DAGs merely to establish statement ownership.
class StatementParents : public ParentStmtAnalysis {
 public:
  void VisitExpr(const ExprPtr&) override {}
};

void Union(ConservativeAccessSet& into, const ConservativeAccessSet& from) {
  into.insert(from.begin(), from.end());
}

MemorySpace Space(const ExprPtr& operand) {
  if (As<TensorType>(operand->GetType())) return MemorySpace::DDR;
  if (const auto buffer = As<BufferType>(operand->GetType())) return buffer->memory_space_;
  CHECK_SPAN(false, operand->span_)
      << "Conservative synchronization: data operand has no qualified storage space";
  return MemorySpace::DDR;
}

// Each catalog entry is a conservative connected component of overlapping
// enclosures. Unknown effects are wildcards, not disjoint unknown roots.
struct RawEffect {
  MemorySpace space;
  PipeType pipe;
  bool write;
  std::optional<backend::BufferSyncByteWindow> enclosure;
  ExprPtr original_root;
};

class StorageClasses {
 public:
  void Build(const std::vector<std::vector<RawEffect>>& sites) {
    std::map<MemorySpace, std::vector<backend::BufferSyncByteWindow>> intervals;
    for (const auto& effects : sites)
      for (const auto& effect : effects)
        if (effect.enclosure) intervals[effect.space].push_back(*effect.enclosure);
    for (auto& [space, windows] : intervals) {
      std::sort(windows.begin(), windows.end(), [](const auto& left, const auto& right) {
        return std::tie(left.begin, left.end) < std::tie(right.begin, right.end);
      });
      for (const auto& window : windows) {
        auto& components = local_[space];
        if (!components.empty() && window.begin < components.back().first.end) {
          auto& component = components.back();
          component.first.end = std::max(component.first.end, window.end);
          domains[component.second - 1].enclosure = component.first;
        } else {
          const auto id = domains.size() + 1;
          components.emplace_back(window, id);
          domains.push_back({id, space, window, nullptr});
        }
      }
    }
    // Original lexical effect order gives deterministic IDs, independent of
    // pointer ordering and hash iteration. Roots are retained by the catalog.
    for (const auto& effects : sites)
      for (const auto& effect : effects)
        if (effect.original_root && !roots_.count(effect.original_root.get())) {
          const auto id = domains.size() + 1;
          roots_.emplace(effect.original_root.get(), id);
          domains.push_back({id, effect.space, {}, effect.original_root});
        }
  }
  size_t Domain(const RawEffect& effect) const {
    if (effect.original_root) return roots_.at(effect.original_root.get());
    if (!effect.enclosure) return 0;
    const auto& components = local_.at(effect.space);
    auto found =
        std::upper_bound(components.begin(), components.end(), effect.enclosure->begin,
                         [](int64_t begin, const auto& component) { return begin < component.first.begin; });
    INTERNAL_CHECK(found != components.begin()) << "Internal error: missing conservative storage component";
    --found;
    INTERNAL_CHECK(effect.enclosure->end <= found->first.end)
        << "Internal error: conservative storage component does not enclose an effect";
    return found->second;
  }
  std::vector<ConservativeStorageDomain> domains;

 private:
  std::map<MemorySpace, std::vector<std::pair<backend::BufferSyncByteWindow, size_t>>> local_;
  std::unordered_map<const Expr*, size_t> roots_;
};

// Summarize possible preceding reads and writes by source pipe. A wildcard
// consumer queries all domains; a known consumer queries its own and wildcard.
// This avoids comparing every incoming fact with every consumer fact.
class ConflictSources {
 public:
  explicit ConflictSources(ConservativeSyncCost& cost) : cost_(&cost) {}
  void Add(const ConservativeAccessFact& fact) {
    const auto& [space, pipe, write, domain] = fact;
    ++cost_->conflict_summary_facts;
    auto& exact = indexed_[{space, domain}];
    (write ? exact.writes : exact.reads).insert(pipe);
    auto& all = all_[space];
    (write ? all.writes : all.reads).insert(pipe);
  }
  std::set<PipeType> Query(const ConservativeAccessFact& consumer) {
    ++cost_->conflict_queries;
    const auto& [space, destination, write, domain] = consumer;
    (void)destination;
    std::set<PipeType> result;
    const auto add = [&](const Summary& summary) {
      result.insert(summary.writes.begin(), summary.writes.end());
      if (write) result.insert(summary.reads.begin(), summary.reads.end());
    };
    if (!domain) {
      if (const auto found = all_.find(space); found != all_.end()) {
        add(found->second);
      }
    } else {
      for (const auto candidate : {size_t{0}, domain}) {
        if (const auto found = indexed_.find({space, candidate}); found != indexed_.end()) {
          add(found->second);
        }
      }
    }
    return result;
  }

 private:
  struct Summary {
    std::set<PipeType> reads, writes;
  };
  ConservativeSyncCost* cost_;
  std::map<std::pair<MemorySpace, size_t>, Summary> indexed_;
  std::map<MemorySpace, Summary> all_;
};

// Payload-completion knowledge only. These facts do not claim that a complete
// logical command prefix (including arbitrary waits) is interchangeable with
// another prefix. Epoch zero denotes unknown incoming payload work, and is not
// initially established on any destination.
class PayloadCompletions {
 public:
  void Reset() {
    issued_.clear();
    established_.clear();
  }
  bool Covers(PipeType source, PipeType destination) const {
    const auto receiver = established_.find(destination);
    if (receiver == established_.end()) return false;
    const auto known = receiver->second.find(source);
    return known != receiver->second.end() && known->second >= Issued(source);
  }
  // Select an inclusion-minimal cover under the facts available at this cut.
  // Distinct source descriptors occur once and have the same destination. No
  // payload executes between them, so their advertised initial completions
  // determine the destination's final knowledge. Removal is sequential: two
  // mutually covering descriptors may not both disappear.
  std::vector<PipeType> Select(const std::vector<PipeType>& required, PipeType destination) const {
    const size_t count = required.size();
    std::vector<bool> base(count);
    std::vector<std::vector<bool>> coverage(count, std::vector<bool>(count));
    for (size_t requirement = 0; requirement < count; ++requirement) {
      base[requirement] = Covers(required[requirement], destination);
      for (size_t candidate = 0; candidate < count; ++candidate)
        coverage[candidate][requirement] = required[candidate] == required[requirement] ||
                                           Covers(required[requirement], required[candidate]);
    }
    std::vector<PipeType> selected;
    for (const auto candidate : SelectConservativeCompletionCover(base, coverage))
      selected.push_back(required[candidate]);
    return selected;
  }
  void Service(PipeType source, PipeType destination) {
    if (source == destination) {
      established_[source][source] = Issued(source);
      return;
    }
    // Forward WAIT imports source payload completion and facts already known
    // on that source. The return WAIT additionally establishes destination
    // payload completion on SOURCE. Issuing the return SET does not establish
    // destination's own payload completion on destination.
    Import(destination, source);
    established_[destination][source] = Issued(source);
    Import(source, destination);
    established_[source][destination] = Issued(destination);
  }
  void Issue(PipeType pipe) { ++issued_[pipe]; }

 private:
  size_t Issued(PipeType pipe) const {
    const auto found = issued_.find(pipe);
    return found == issued_.end() ? 0 : found->second;
  }
  void Import(PipeType destination, PipeType source) {
    const auto sender = established_.find(source);
    if (sender == established_.end()) return;
    auto& receiver = established_[destination];
    for (const auto& [pipe, epoch] : sender->second) {
      auto [known, inserted] = receiver.emplace(pipe, epoch);
      if (!inserted) known->second = std::max(known->second, epoch);
    }
  }
  std::map<PipeType, size_t> issued_;
  std::map<PipeType, std::map<PipeType, size_t>> established_;
};

struct FlowState {
  explicit FlowState(ConservativeSyncCost& cost) : sources(cost) {}
  void Add(const ConservativeAccessSet& added) {
    for (const auto& fact : added)
      if (effects.insert(fact).second) sources.Add(fact);
  }
  ConservativeAccessSet effects;
  ConflictSources sources;
  PayloadCompletions completions;
};

struct End {
  std::optional<size_t> yield;
  bool returned = false;
  bool Terminal() const { return yield.has_value() || returned; }
};

class Builder {
 public:
  Builder(const FunctionPtr& function, std::string target,
          const std::vector<std::shared_ptr<const NativeSyncChild>>& retained,
          const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful,
          bool retain_incoming_diagnostics)
      : function_(function),
        target_(std::move(target)),
        placement_(function),
        retain_incoming_diagnostics_(retain_incoming_diagnostics) {
    const auto* context = PassContext::Current();
    disjoint_gm_ = context && context->GetAutoSyncDisjointGM();
    scopes_.emplace_back();
    for (const auto& parameter : function->params_) {
      const auto& type = parameter->GetType();
      CHECK_SPAN(
          As<ScalarType>(type) || As<TensorType>(type) || As<BufferType>(type) || As<MultiBufferType>(type),
          parameter->span_)
          << "Conservative synchronization: parameter requires an invocation ownership/readiness contract";
      Bind(parameter);
      if (const auto tensor = As<TensorType>(type)) {
        const auto memory = tensor->memref_.value_or(nullptr);
        gm_roots_.emplace(parameter.get(), memory ? ExprPtr(memory->base_) : ExprPtr(parameter));
      }
    }
    for (const auto& child : retained) {
      CHECK(child) << "Conservative synchronization: missing retained child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), child, {}, {}, {}, {}, {}});
    }
    for (const auto& child : eventful) {
      CHECK(child) << "Conservative synchronization: missing eventful child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), {}, {}, {}, {}, child, {}});
    }
  }

  void Build() {
    Read(function_->body_);
    RefineStorage();
    RebuildGenerated(function_->body_);
    for (auto& child : children) {
      const auto found = generated_.find(child.scope.get());
      CHECK(found != generated_.end()) << "Retained child is outside the original semantic input";
      child.effects = found->second;
    }
  }

  std::vector<ConservativeSyncSite> PrimitiveSites() {
    std::vector<ConservativeSyncSite> result;
    for (size_t index = 0; index < sites.size(); ++index)
      if (!inside_child_[index]) result.push_back(std::move(sites[index]));
    return result;
  }

  ConservativeAccessSet Solve() {
    auto solved = Flow(function_->body_, FlowState(cost));
    PayloadCompletions certificate;
    Certify(function_->body_, certificate);
    INTERNAL_CHECK(cost.certificate_omissions == cost.suppressed_services)
        << "Internal error: conservative service certificate omitted a proposed deletion";
    return std::move(solved.effects);
  }

  FlowState Flow(const StmtPtr& statement, FlowState incoming) {
    ++cost.dataflow_visits;
    if (const auto found = child_index_.find(statement.get()); found != child_index_.end()) {
      auto& entry = children[found->second];
      incoming.completions.Reset();
      if (retain_incoming_diagnostics_) {
        entry.incoming = incoming.effects;
        cost.incoming_diagnostic_facts += entry.incoming.size();
      }
      std::set<std::pair<PipeType, PipeType>> services;
      for (const auto& effect : entry.effects)
        for (const auto source : incoming.sources.Query(effect))
          services.emplace(source, std::get<1>(effect));
      if (entry.eventful) {
        // Explicit selected-token entry acknowledgments rearm every child
        // direction across arbitrary skipped/reset visits. Closed matching at
        // child exit alone would not establish this causal path.
        for (const auto source : entry.eventful->Pipes())
          for (const auto destination : entry.eventful->Pipes())
            if (source != destination) services.emplace(source, destination);
      }
      entry.services.assign(services.begin(), services.end());
      entry.required_services = entry.services;
      cost.baseline_services += services.size();
      cost.services += services.size();

      // Empty notification state does not complete any payload. All child
      // effects remain possibly pending, including across zero/skipped visits.
      incoming.Add(entry.effects);
      incoming.completions.Reset();
      return incoming;
    }
    if (const auto seq = As<SeqStmts>(statement)) {
      for (const auto& child : seq->stmts_) incoming = Flow(child, std::move(incoming));
    } else if (const auto branch = As<IfStmt>(statement)) {
      incoming.completions.Reset();
      cost.control_snapshot_facts += incoming.effects.size();
      auto then_exit = Flow(branch->then_body_, incoming);
      if (branch->else_body_) {
        auto else_exit = Flow(*branch->else_body_, std::move(incoming));
        cost.control_merge_facts += else_exit.effects.size();
        then_exit.Add(else_exit.effects);
      } else {
        cost.control_merge_facts += incoming.effects.size();
        then_exit.Add(incoming.effects);
      }
      incoming = std::move(then_exit);
      incoming.completions.Reset();
    } else if (const auto loop = As<ForStmt>(statement)) {
      incoming.Add(generated_.at(loop->body_.get()));
      incoming.completions.Reset();
      incoming = Flow(loop->body_, std::move(incoming));
      incoming.completions.Reset();
    } else if (const auto loop = As<WhileStmt>(statement)) {
      incoming.Add(generated_.at(loop->body_.get()));
      incoming.completions.Reset();
      incoming = Flow(loop->body_, std::move(incoming));
      incoming.completions.Reset();
    } else if (const auto found = site_index_.find(statement.get()); found != site_index_.end()) {
      auto& site = sites[found->second];
      if (retain_incoming_diagnostics_) {
        site.incoming = incoming.effects;
        cost.incoming_diagnostic_facts += site.incoming.size();
      }
      std::set<PipeType> sources;
      for (const auto& effect : site.effects) {
        const auto conflicting = incoming.sources.Query(effect);
        sources.insert(conflicting.begin(), conflicting.end());
      }
      site.required_source_pipes.assign(sources.begin(), sources.end());
      cost.baseline_services += sources.size();
      const auto destination = site.operation.Implementation().Pipe();
      site.source_pipes = incoming.completions.Select(site.required_source_pipes, destination);
      cost.suppressed_services += site.required_source_pipes.size() - site.source_pipes.size();
      cost.services += site.source_pipes.size();
      // Replay only the retained descriptors. Deleted acknowledgments cannot
      // establish source-side facts used by another consumer cut.
      for (const auto source : site.source_pipes) incoming.completions.Service(source, destination);
      incoming.completions.Issue(destination);
      incoming.Add(site.effects);
    }
    return incoming;
  }

  std::vector<ConservativeSyncSite> sites;
  std::vector<ConservativeSyncChildEntry> children;
  ConservativeSyncCost cost;
  std::vector<ConservativeStorageDomain> domains;
  bool disjoint_gm_ = false;

 private:
  // Verify the complete immutable selection over original cuts and guards.
  // Required vectors come from the no-KILL storage analysis; only a retained
  // service can establish facts used by later omissions. This pass never adds
  // commands, changes an assignment, or inspects an emitted physical program.
  void Certify(const StmtPtr& statement, PayloadCompletions& known) {
    ++cost.certificate_visits;
    if (const auto child = child_index_.find(statement.get()); child != child_index_.end()) {
      const auto& entry = children[child->second];
      INTERNAL_CHECK(entry.services == entry.required_services)
          << "Internal error: protected child/rearming services changed";
      known.Reset();
      return;
    }
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      const auto& selected = sites[site->second];
      size_t next = 0;
      const auto destination = selected.operation.Implementation().Pipe();
      for (const auto source : selected.required_source_pipes) {
        if (next < selected.source_pipes.size() && selected.source_pipes[next] == source) {
          known.Service(source, destination);
          ++next;
        }
      }
      INTERNAL_CHECK(next == selected.source_pipes.size())
          << "Internal error: conservative service selection added or reordered a service";
      // Every requested completion must hold at the payload start. It may be
      // supplied by a later retained descriptor at this SAME consumer cut.
      for (const auto source : selected.required_source_pipes)
        INTERNAL_CHECK(known.Covers(source, destination))
            << "Internal error: conservative service selection omits a payload prerequisite";
      cost.certificate_omissions += selected.required_source_pipes.size() - selected.source_pipes.size();
      known.Issue(destination);
      return;
    }
    if (const auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Certify(child, known);
    } else if (const auto branch = As<IfStmt>(statement)) {
      known.Reset();
      Certify(branch->then_body_, known);
      known.Reset();
      if (branch->else_body_) Certify(*branch->else_body_, known);
      known.Reset();
    } else if (const auto loop = As<ForStmt>(statement)) {
      known.Reset();
      Certify(loop->body_, known);
      known.Reset();
    } else if (const auto loop = As<WhileStmt>(statement)) {
      known.Reset();
      Certify(loop->body_, known);
      known.Reset();
    }
  }

  void RefineStorage() {
    std::vector<std::vector<RawEffect>> raw(sites.size());
    for (size_t site = 0; site < sites.size(); ++site) {
      const auto& operation = sites[site].operation;
      const auto& call = operation.Operation();
      const auto pipe = operation.Implementation().Pipe();
      const auto append = [&](MemorySpace space, BufferAccess mode,
                              std::optional<backend::BufferSyncByteWindow> enclosure, ExprPtr root) {
        if (mode == BufferAccess::Read || mode == BufferAccess::ReadWrite)
          raw[site].push_back({space, pipe, false, enclosure, root});
        if (mode == BufferAccess::Write || mode == BufferAccess::ReadWrite)
          raw[site].push_back({space, pipe, true, enclosure, root});
      };
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        const auto mode = operation.ArgumentEffects()[index].data;
        if (mode == BufferAccess::None) continue;
        const auto& operand = call->args_[index];
        const auto space = Space(operand);
        std::optional<backend::BufferSyncByteWindow> enclosure;
        ExprPtr root;
        if (space == MemorySpace::DDR) {
          if (disjoint_gm_) {
            if (const auto found = gm_roots_.find(operand.get()); found != gm_roots_.end())
              root = found->second;
          }
        } else {
          ++cost.enclosure_lookups;
          enclosure = placement_.Window(operand);
        }
        append(space, mode, enclosure, std::move(root));
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        std::optional<backend::BufferSyncByteWindow> enclosure;
        if (effect.space != MemorySpace::DDR && effect.begin >= 0 && effect.end > effect.begin)
          enclosure = backend::BufferSyncByteWindow{effect.begin, effect.end};
        append(effect.space, effect.data, enclosure, nullptr);
      }
    }
    StorageClasses classes;
    classes.Build(raw);
    for (size_t site = 0; site < sites.size(); ++site) {
      auto& effects = sites[site].effects;
      effects.clear();
      for (const auto& effect : raw[site])
        effects.emplace(effect.space, effect.pipe, effect.write, classes.Domain(effect));
    }
    domains = std::move(classes.domains);
    cost.storage_domains = domains.size();
  }

  const ConservativeAccessSet& RebuildGenerated(const StmtPtr& statement) {
    auto& summary = generated_.at(statement.get());
    summary.clear();
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      summary = sites[site->second].effects;
    } else if (const auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Union(summary, RebuildGenerated(child));
    } else if (const auto branch = As<IfStmt>(statement)) {
      Union(summary, RebuildGenerated(branch->then_body_));
      if (branch->else_body_) Union(summary, RebuildGenerated(*branch->else_body_));
    } else if (const auto loop = As<ForStmt>(statement)) {
      Union(summary, RebuildGenerated(loop->body_));
    } else if (const auto loop = As<WhileStmt>(statement)) {
      Union(summary, RebuildGenerated(loop->body_));
    }
    return summary;
  }

  void Bind(const VarPtr& variable) {
    CHECK_SPAN(ready_.insert(variable.get()).second, variable->span_)
        << "Conservative synchronization: a ready SSA definition is redefined";
    scopes_.back().push_back(variable.get());
  }
  void PopScope() {
    for (const auto* variable : scopes_.back()) ready_.erase(variable);
    scopes_.pop_back();
  }

  void Ready(const ExprPtr& expression) {
    cost.expression_visits += CheckSyncScalarReadiness(
        expression, [this](const VarPtr& variable) { return ready_.count(variable.get()) != 0; });
  }

  void Payload(const StmtPtr& cut, const CallPtr& call, const VarPtr& result) {
    CHECK_SPAN(call->attrs_.empty(), call->span_)
        << "Conservative synchronization: operation attributes require an execution contract";
    for (const auto& argument : call->args_) Ready(argument);
    if (auto descriptor = backend::QualifyBufferSyncNonPayload(call, target_)) {
      const auto behavior = descriptor->Result().behavior;
      CHECK_SPAN(static_cast<bool>(result) == (behavior != BufferResultBehavior::None), call->span_)
          << "Conservative synchronization: descriptor result must have an original SSA binding";
      if (result) {
        Bind(result);
        const auto& ownership = descriptor->Result();
        if (ownership.behavior == BufferResultBehavior::Alias && ownership.alias_arg) {
          const auto found = gm_roots_.find(call->args_[*ownership.alias_arg].get());
          if (found != gm_roots_.end()) gm_roots_.emplace(result.get(), found->second);
        }
      }
      return;
    }
    auto operation = backend::QualifyBufferSyncOperation(call, function_->func_type_, target_);
    placement_.Require(operation);
    const bool ready_result = operation.Implementation().ProducesReadyScalar();
    CHECK_SPAN((!result && operation.Result().behavior == BufferResultBehavior::None) ||
                   (result && ready_result && operation.Result().behavior == BufferResultBehavior::Value),
               call->span_)
        << "Conservative synchronization: payload-produced scalar/handle readiness is unavailable";
    ConservativeAccessSet effects;
    for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
      const auto effect = operation.ArgumentEffects()[index].data;
      if (effect == BufferAccess::None) continue;
      const auto space = Space(call->args_[index]);
      const auto pipe = operation.Implementation().Pipe();
      if (effect == BufferAccess::Read || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, false, 0);
      if (effect == BufferAccess::Write || effect == BufferAccess::ReadWrite)
        effects.emplace(space, pipe, true, 0);
    }
    for (const auto& effect : operation.Implementation().ImplicitEffects()) {
      const auto pipe = operation.Implementation().Pipe();
      if (effect.data == BufferAccess::Read || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, false, 0);
      if (effect.data == BufferAccess::Write || effect.data == BufferAccess::ReadWrite)
        effects.emplace(effect.space, pipe, true, 0);
    }
    CHECK_SPAN(!effects.empty(), call->span_)
        << "Conservative synchronization: command-only payload requires a protocol contract";
    const auto inserted = site_index_.emplace(cut.get(), sites.size()).second;
    CHECK_SPAN(inserted, cut->span_)
        << "Conservative synchronization: shared statement identity has ambiguous insertion cuts";
    generated_[cut.get()] = effects;
    sites.push_back({cut, std::move(operation), std::move(effects), {}, {}, {}});
    inside_child_.push_back(native_child_depth_ != 0);
    // The ordinary service protects the scalar instruction's storage read.
    // Only its selected blocking native return then makes the original SSA
    // available to following addresses/control; it is never a pure query.
    if (ready_result) Bind(result);
  }

  End Read(const StmtPtr& statement) {
    ++cost.statements;
    generated_.try_emplace(statement.get());
    if (const auto seq = As<SeqStmts>(statement)) {
      End end;
      for (const auto& child : seq->stmts_) {
        CHECK_SPAN(!end.Terminal(), child->span_)
            << "Conservative synchronization: statements after a terminal cut are unavailable";
        end = Read(child);
        Union(generated_.at(statement.get()), generated_.at(child.get()));
      }
      return end;
    }
    if (const auto assignment = As<AssignStmt>(statement)) {
      if (const auto call = As<Call>(assignment->value_)) {
        Payload(statement, call, assignment->var_);
      } else {
        Ready(assignment->value_);
        Bind(assignment->var_);
        if (const auto found = gm_roots_.find(assignment->value_.get()); found != gm_roots_.end())
          gm_roots_.emplace(assignment->var_.get(), found->second);
      }
      return {};
    }
    if (const auto eval = As<EvalStmt>(statement)) {
      if (const auto call = As<Call>(eval->expr_))
        Payload(statement, call, nullptr);
      else
        Ready(eval->expr_);
      return {};
    }
    if (const auto branch = As<IfStmt>(statement)) {
      Ready(branch->condition_);
      ++control_depth_;
      scopes_.emplace_back();
      const auto then_end = Read(branch->then_body_);
      PopScope();
      End else_end;
      if (branch->else_body_) {
        scopes_.emplace_back();
        else_end = Read(*branch->else_body_);
        PopScope();
        Union(generated_.at(statement.get()), generated_.at(branch->else_body_->get()));
      }
      --control_depth_;
      CHECK_SPAN(!then_end.returned && !else_end.returned, statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      const auto count = branch->return_vars_.size();
      CHECK_SPAN(then_end.yield.value_or(0) == count && else_end.yield.value_or(0) == count, statement->span_)
          << "Conservative synchronization: branch result/yield arity mismatch";
      for (const auto& variable : branch->return_vars_) Bind(variable);
      Union(generated_.at(statement.get()), generated_.at(branch->then_body_.get()));
      return {};
    }
    if (const auto loop = As<ForStmt>(statement)) {
      CHECK_SPAN(loop->kind_ != ForKind::Parallel, statement->span_)
          << "Conservative synchronization: concurrent loop execution requires another domain contract";
      Ready(loop->start_);
      Ready(loop->stop_);
      Ready(loop->step_);
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, loop->loop_var_, nullptr);
      return {};
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      ReadLoop(statement, loop->body_, loop->iter_args_, loop->return_vars_, nullptr, loop->condition_);
      return {};
    }
    if (const auto yield = As<YieldStmt>(statement)) {
      for (const auto& value : yield->value_) Ready(value);
      return {yield->value_.size(), false};
    }
    if (const auto ret = As<ReturnStmt>(statement)) {
      CHECK_SPAN(control_depth_ == 0, statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      for (const auto& value : ret->value_) Ready(value);
      return {std::nullopt, true};
    }
    CHECK_SPAN(false, statement->span_)
        << "Conservative synchronization: unsupported structured control or command statement: "
        << statement->TypeName();
    return {};
  }

  void ReadLoop(const StmtPtr& loop, const StmtPtr& body, const std::vector<IterArgPtr>& arguments,
                const std::vector<VarPtr>& results, const VarPtr& index, const ExprPtr& condition) {
    CHECK_SPAN(arguments.size() == results.size(), loop->span_)
        << "Conservative synchronization: loop result/initializer arity mismatch";
    for (const auto& argument : arguments) Ready(argument->initValue_);
    scopes_.emplace_back();
    if (index) Bind(index);
    for (const auto& argument : arguments) Bind(argument);
    if (condition) Ready(condition);
    const bool child = child_index_.count(loop.get()) != 0;
    CHECK(!child || native_child_depth_ == 0)
        << "Conservative synchronization: retained child scopes must not overlap";
    if (child) ++native_child_depth_;
    ++control_depth_;
    const auto end = Read(body);
    --control_depth_;
    if (child) --native_child_depth_;
    CHECK_SPAN(!end.returned && end.yield.value_or(0) == arguments.size(), body->span_)
        << "Conservative synchronization: loop backedge lacks matching ready yields";
    PopScope();
    for (const auto& result : results) Bind(result);
    Union(generated_.at(loop.get()), generated_.at(body.get()));
  }

  FunctionPtr function_;
  std::string target_;
  NativeSyncPlacement placement_;
  bool retain_incoming_diagnostics_;
  std::unordered_map<const Expr*, ExprPtr> gm_roots_;
  std::vector<std::vector<const Var*>> scopes_;
  std::unordered_set<const Var*> ready_;
  size_t control_depth_ = 0;
  size_t native_child_depth_ = 0;
  std::vector<bool> inside_child_;
  std::unordered_map<const Stmt*, size_t> child_index_;
  std::unordered_map<const Stmt*, ConservativeAccessSet> generated_;
  std::unordered_map<const Stmt*, size_t> site_index_;
};
}  // namespace

std::vector<StmtPtr> ConservativeSyncAnalysis::ControlPath(size_t site) const {
  CHECK(site < sites_.size()) << "Conservative synchronization: invalid diagnostic site index";
  std::vector<StmtPtr> path;
  for (auto parent = parents_->GetParent(sites_[site].cut); parent; parent = parents_->GetParent(parent)) {
    if (As<IfStmt>(parent) || As<ForStmt>(parent) || As<WhileStmt>(parent)) path.push_back(parent);
  }
  std::reverse(path.begin(), path.end());
  return path;
}

ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target,
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children,
    const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful_children,
    bool retain_incoming_diagnostics, bool automatic_children) {
  CHECK(function && function->ir_stage_ == FunctionIRStage::Buffer)
      << "Conservative synchronization requires Buffer IR";
  CHECK(function->func_type_ == FunctionType::AIC || function->func_type_ == FunctionType::AIV)
      << "Conservative synchronization requires an owned local AIC/AIV domain";
  backend::RequireLocalSyncExecutionContract(target);
  RequireBufferRuntimeValid(function);
  CHECK(!automatic_children || (children.empty() && eventful_children.empty()))
      << "Automatic child selection cannot replace explicitly supplied child capabilities";
  AutomaticSyncChildren selection;
  std::vector<std::shared_ptr<const EventfulSyncChild>> selected = eventful_children;
  if (automatic_children) {
    selection = PrepareAutomaticSyncChildren(function, target);
    for (const auto& attempt : selection.attempts)
      if (attempt.child) selected.push_back(attempt.child);
  }
  Builder builder(function, target, children, selected, retain_incoming_diagnostics);
  builder.Build();
  auto exit = builder.Solve();
  builder.cost.retained_facts = exit.size();
  for (const auto& site : builder.sites)
    builder.cost.retained_facts += site.effects.size() + site.incoming.size();
  for (const auto& child : builder.children)
    builder.cost.retained_facts += child.effects.size() + child.incoming.size();
  auto parents = std::make_shared<StatementParents>();
  parents->BuildMap(function);
  builder.cost.parent_links = builder.cost.statements ? builder.cost.statements - 1 : 0;
  return ConservativeSyncAnalysis(function, builder.PrimitiveSites(), std::move(exit), builder.cost,
                                  std::move(parents), std::move(builder.children), std::move(builder.domains),
                                  retain_incoming_diagnostics, builder.disjoint_gm_, std::move(selection));
}
}  // namespace pypto::ir::sync
