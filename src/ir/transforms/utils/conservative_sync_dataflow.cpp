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
#include <array>
#include <deque>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
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
  bool candidate_full_overwrite = false;
  std::optional<backend::BufferSyncByteWindow> exact_full_window;
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

bool IsFullOverwriteOperand(const CallPtr& call, BufferAccess mode, MemorySpace space, const ExprPtr& operand) {
  if (space == MemorySpace::DDR) return false;
  const std::string& name = call->op_->name_;
  if (name == "buffer.load") {
    if (mode != BufferAccess::Write) return false;
    const auto buf = As<BufferType>(operand->GetType());
    return buf && buf->shape_.size() >= 2;
  }
  if (name == "buffer.copy" || name == "buffer.extract" || name == "buffer.matmul") {
    return mode == BufferAccess::Write;
  }
  if (name == "buffer.matmul_acc") {
    return mode == BufferAccess::ReadWrite;
  }
  if (backend::FindBufferElementwiseRecipe(name) || backend::FindBufferConversionRecipe(name)) {
    return mode == BufferAccess::Write;
  }
  return false;
}

constexpr size_t kAllPipeLaneCount = kPhysicalPipeCount + 1;  // MTE1..FIX + ALL.

PipeMask LaneToMask(PipeType pipe) {
  const auto index = static_cast<size_t>(pipe);
  INTERNAL_CHECK(index < kAllPipeLaneCount) << "Internal error: invalid pipeline lane";
  return PipeMask{1} << index;
}

void AppendPipeMaskToSet(PipeMask mask, std::set<PipeType>& out) {
  for (size_t index = 0; index < kAllPipeLaneCount; ++index) {
    if ((mask & (PipeMask{1} << index)) != 0) out.insert(static_cast<PipeType>(index));
  }
}

size_t PopCount(PipeMask mask) {
  size_t count = 0;
  for (; mask != 0; mask &= (mask - 1)) ++count;
  return count;
}

// Pass 1 monotone state: tracks live (not cleanly overwritten) read/write pipes per
// (MemorySpace, domain). Domain 0 represents unknown/wildcard storage within `space`.
class LiveStorageState {
 public:
  struct DomainLanes {
    PipeMask reads = 0;
    PipeMask writes = 0;
    bool operator==(const DomainLanes& other) const {
      return reads == other.reads && writes == other.writes;
    }
    bool operator!=(const DomainLanes& other) const { return !(*this == other); }
  };

  bool MergeFrom(const LiveStorageState& other) {
    bool changed = false;
    for (const auto& [key, lanes] : other.entries_) {
      auto& dst = entries_[key];
      const DomainLanes before = dst;
      dst.reads |= lanes.reads;
      dst.writes |= lanes.writes;
      if (dst != before) changed = true;
    }
    return changed;
  }

  size_t KillDomain(MemorySpace space, size_t domain) {
    if (domain == 0) return 0;  // Wildcard domain 0 cannot be safely killed.
    const auto found = entries_.find({space, domain});
    if (found == entries_.end()) return 0;
    const size_t killed = PopCount(found->second.reads) + PopCount(found->second.writes);
    entries_.erase(found);
    return killed;
  }

  size_t CountDomain(MemorySpace space, size_t domain) const {
    const auto found = entries_.find({space, domain});
    if (found == entries_.end()) return 0;
    return PopCount(found->second.reads) + PopCount(found->second.writes);
  }

  void AddEffects(const ConservativeAccessSet& added) {
    for (const auto& fact : added) {
      const auto& [space, pipe, write, domain] = fact;
      auto& dst = entries_[{space, domain}];
      const PipeMask bit = LaneToMask(pipe);
      if (write) {
        dst.writes |= bit;
      } else {
        dst.reads |= bit;
      }
    }
  }

  std::set<PipeType> QueryConflicts(const ConservativeAccessFact& consumer, ConservativeSyncCost& cost) const {
    ++cost.conflict_queries;
    const auto& [space, destination, write, domain] = consumer;
    (void)destination;
    std::set<PipeType> result;
    const auto collect = [&](const DomainLanes& lanes) {
      AppendPipeMaskToSet(lanes.writes, result);
      if (write) AppendPipeMaskToSet(lanes.reads, result);
    };
    if (domain == 0) {
      auto it = entries_.lower_bound({space, 0});
      for (; it != entries_.end() && it->first.first == space; ++it) collect(it->second);
    } else {
      for (const size_t candidate : {size_t{0}, domain}) {
        if (const auto found = entries_.find({space, candidate}); found != entries_.end()) {
          collect(found->second);
        }
      }
    }
    return result;
  }

  ConservativeAccessSet ToAccessSet() const {
    ConservativeAccessSet result;
    for (const auto& [key, lanes] : entries_) {
      const auto& [space, domain] = key;
      for (size_t index = 0; index < kAllPipeLaneCount; ++index) {
        const PipeMask bit = PipeMask{1} << index;
        const auto pipe = static_cast<PipeType>(index);
        if ((lanes.reads & bit) != 0) result.emplace(space, pipe, false, domain);
        if ((lanes.writes & bit) != 0) result.emplace(space, pipe, true, domain);
      }
    }
    return result;
  }

  size_t FactCount() const {
    size_t total = 0;
    for (const auto& [key, lanes] : entries_) {
      (void)key;
      total += PopCount(lanes.reads) + PopCount(lanes.writes);
    }
    return total;
  }

  bool operator!=(const LiveStorageState& other) const { return entries_ != other.entries_; }

 private:
  std::map<std::pair<MemorySpace, size_t>, DomainLanes> entries_;
};

// Pass 2 sparse delivery-debt lattice (Draft Section 6.4.2-6.4.3).
// Tracks pending completion masks per (space, domain, write, origin_pipe). A set bit `d`
// indicates that the event on `origin_pipe` is not yet guaranteed ordered before pipe `d`.
// Absent entries carry zero pending debts.
struct DebtKey {
  MemorySpace space;
  size_t domain;
  bool write;
  PipeType src;

  bool operator<(const DebtKey& other) const {
    return std::tie(space, domain, write, src) < std::tie(other.space, other.domain, other.write, other.src);
  }
  bool operator==(const DebtKey& other) const {
    return space == other.space && domain == other.domain && write == other.write && src == other.src;
  }
};

class DebtState {
 public:
  bool MergeFrom(const DebtState& other) {
    bool changed = false;
    for (const auto& [key, mask] : other.pending_) {
      auto& dst = pending_[key];
      const PipeMask merged = dst | mask;
      if (merged != dst) {
        dst = merged;
        changed = true;
      }
    }
    return changed;
  }

  // Ordered completion relay step `a -> b`. `a` itself is ready for events originated by `a`
  // or already delivered to `a` on all reaching paths.
  void DeliverOneWay(PipeType src, PipeType dst) {
    const PipeMask src_mask = PipeToMask(src);
    const PipeMask dst_mask = PipeToMask(dst);
    if (src_mask == 0 || dst_mask == 0) return;
    for (auto it = pending_.begin(); it != pending_.end();) {
      const bool src_ready = (it->first.src == src) || ((it->second & src_mask) == 0);
      if (src_ready) it->second &= ~dst_mask;
      if (it->second == 0) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Four-command hardware sync `(src, dst)`: forward WAIT then return WAIT.
  void DeliverService(PipeType src, PipeType dst) {
    if (src == dst) {
      DeliverOneWay(src, dst);
      return;
    }
    DeliverOneWay(src, dst);
    DeliverOneWay(dst, src);
  }

  void ApplySummary(const CompletionSummary& summary) {
    for (auto it = pending_.begin(); it != pending_.end();) {
      PipeMask next = it->second;
      for (size_t mid = 0; mid < kPhysicalPipeCount; ++mid) {
        const auto mid_pipe = static_cast<PipeType>(mid);
        const PipeMask mid_bit = PipeMask{1} << mid;
        const bool mid_ready = (it->first.src == mid_pipe) || ((it->second & mid_bit) == 0);
        if (mid_ready) {
          // Strict cross-pipe relays (`q != mid`) propagate already-established readiness.
          const PipeMask strict_relay = summary.relay[mid] & ~mid_bit;
          next &= ~strict_relay;
        }
      }
      if (next == 0) {
        it = pending_.erase(it);
      } else {
        it->second = next;
        ++it;
      }
    }
  }

  void KillDomain(MemorySpace space, size_t domain) {
    if (domain == 0) return;
    for (auto it = pending_.lower_bound({space, domain, false, PipeType::MTE1});
         it != pending_.end() && it->first.space == space && it->first.domain == domain;) {
      it = pending_.erase(it);
    }
  }

  void AddEffects(const ConservativeAccessSet& added, const CompletionSummary* summary = nullptr) {
    for (const auto& fact : added) {
      const auto& [space, pipe, write, domain] = fact;
      if (pipe == PipeType::ALL) continue;
      const PipeMask pipe_bit = PipeToMask(pipe);
      PipeMask initial = kAllPhysicalPipesMask;
      if (summary != nullptr) {
        const size_t idx = static_cast<size_t>(pipe);
        if (idx < kPhysicalPipeCount) {
          // Keep self-pipe bit unless intra-pipe ordering is guaranteed; clear strict safe destinations.
          initial &= ~(summary->post_write_safe[idx] & ~pipe_bit);
        }
      }
      if (initial != 0) {
        pending_[{space, domain, write, pipe}] = initial;
      } else {
        pending_.erase({space, domain, write, pipe});
      }
    }
  }

  // Returns true iff every pending debt from `source` that conflicts with `site_effects`
  // is already guaranteed complete at `probe`.
  bool CoversConflicts(PipeType source, PipeType probe, const ConservativeAccessSet& site_effects) const {
    const PipeMask probe_mask = PipeToMask(probe);
    if (probe_mask == 0) return false;
    for (const auto& [key, mask] : pending_) {
      if (key.src != source) continue;
      if (!ConflictsAny(key, site_effects)) continue;
      if ((mask & probe_mask) != 0) return false;
    }
    return true;
  }

  std::vector<PipeType> Select(const std::vector<PipeType>& required, PipeType destination,
                               const ConservativeAccessSet& site_effects) const {
    const size_t count = required.size();
    std::vector<bool> base(count);
    std::vector<std::vector<bool>> coverage(count, std::vector<bool>(count));
    for (size_t requirement = 0; requirement < count; ++requirement) {
      base[requirement] = CoversConflicts(required[requirement], destination, site_effects);
      for (size_t candidate = 0; candidate < count; ++candidate) {
        coverage[candidate][requirement] =
            required[candidate] == required[requirement] ||
            CoversConflicts(required[requirement], required[candidate], site_effects);
      }
    }
    std::vector<PipeType> selected;
    for (const auto candidate : SelectConservativeCompletionCover(base, coverage)) {
      selected.push_back(required[candidate]);
    }
    return selected;
  }

  bool operator!=(const DebtState& other) const { return pending_ != other.pending_; }

 private:
  static bool ConflictsAny(const DebtKey& key, const ConservativeAccessSet& site_effects) {
    for (const auto& fact : site_effects) {
      const auto& [c_space, c_pipe, c_write, c_domain] = fact;
      (void)c_pipe;
      if (c_space != key.space) continue;
      if (!key.write && !c_write) continue;
      if (key.domain == 0 || c_domain == 0 || key.domain == c_domain) return true;
    }
    return false;
  }

  std::map<DebtKey, PipeMask> pending_;
};

struct CfgVertex {
  enum class Kind { Nop, Site, Child, BranchMerge };
  Kind kind = Kind::Nop;
  size_t index = 0;
  std::vector<size_t> preds;
  std::vector<size_t> succs;
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
          bool retain_incoming_diagnostics, bool enable_refresh_kill)
      : function_(function),
        target_(std::move(target)),
        placement_(function),
        retain_incoming_diagnostics_(retain_incoming_diagnostics),
        enable_refresh_kill_(enable_refresh_kill) {
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
      children.push_back({child->Scope(), child, {}, {}, {}, {}, {}, CompletionSummary::Identity()});
    }
    for (const auto& child : eventful) {
      CHECK(child) << "Conservative synchronization: missing eventful child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), {}, {}, {}, {}, child, {}, CompletionSummary::Identity()});
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
      child.completion = CompletionSummary::Identity();
      child.completion.has_bypass_path = true;
      for (const auto& fact : child.effects) {
        const PipeType pipe = std::get<1>(fact);
        if (pipe != PipeType::ALL) child.completion.active |= PipeToMask(pipe);
      }
    }
  }

  std::vector<ConservativeSyncSite> PrimitiveSites() {
    std::vector<ConservativeSyncSite> result;
    for (size_t index = 0; index < sites.size(); ++index)
      if (!inside_child_[index]) result.push_back(std::move(sites[index]));
    return result;
  }

  ConservativeAccessSet Solve() {
    std::vector<CfgVertex> cfg;
    const size_t cfg_entry = 0;
    cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
    const size_t body_exit = BuildSubCfg(function_->body_, cfg_entry, cfg);
    const size_t cfg_exit = cfg.size();
    cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
    AddCfgEdge(cfg, body_exit, cfg_exit);

    std::vector<size_t> rpo;
    std::vector<size_t> rpo_pos;
    ComputeRpo(cfg, cfg_entry, rpo, rpo_pos);

    std::vector<LiveStorageState> live_in, live_out;
    SolveLiveStorage(cfg, rpo, rpo_pos, cfg_entry, live_in, live_out);
    ExtractSourcePipeServices(cfg, rpo, live_in, live_out);
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, cfg_entry);

    return live_out[cfg_exit].ToAccessSet();
  }

  std::vector<ConservativeSyncSite> sites;
  std::vector<ConservativeSyncChildEntry> children;
  ConservativeSyncCost cost;
  std::vector<ConservativeStorageDomain> domains;
  bool disjoint_gm_ = false;

 private:
  static void AddCfgEdge(std::vector<CfgVertex>& cfg, size_t from, size_t to) {
    auto& succs = cfg[from].succs;
    if (std::find(succs.begin(), succs.end(), to) == succs.end()) {
      succs.push_back(to);
      cfg[to].preds.push_back(from);
    }
  }

  size_t BuildSubCfg(const StmtPtr& statement, size_t entry, std::vector<CfgVertex>& cfg) const {
    if (const auto child = child_index_.find(statement.get()); child != child_index_.end()) {
      const size_t node = cfg.size();
      cfg.push_back({CfgVertex::Kind::Child, child->second, {}, {}});
      AddCfgEdge(cfg, entry, node);
      return node;
    }
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      const size_t node = cfg.size();
      cfg.push_back({CfgVertex::Kind::Site, site->second, {}, {}});
      AddCfgEdge(cfg, entry, node);
      return node;
    }
    if (const auto seq = As<SeqStmts>(statement)) {
      size_t cur = entry;
      for (const auto& child : seq->stmts_) cur = BuildSubCfg(child, cur, cfg);
      return cur;
    }
    if (const auto branch = As<IfStmt>(statement)) {
      const size_t then_exit = BuildSubCfg(branch->then_body_, entry, cfg);
      const size_t else_exit = branch->else_body_ ? BuildSubCfg(*branch->else_body_, entry, cfg) : entry;
      const size_t merge = cfg.size();
      cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
      AddCfgEdge(cfg, then_exit, merge);
      AddCfgEdge(cfg, else_exit, merge);
      return merge;
    }
    if (const auto loop = As<ForStmt>(statement)) {
      const size_t header = cfg.size();
      cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
      AddCfgEdge(cfg, entry, header);
      const size_t body_exit = BuildSubCfg(loop->body_, header, cfg);
      AddCfgEdge(cfg, body_exit, header);
      return header;
    }
    if (const auto loop = As<WhileStmt>(statement)) {
      const size_t header = cfg.size();
      cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
      AddCfgEdge(cfg, entry, header);
      const size_t body_exit = BuildSubCfg(loop->body_, header, cfg);
      AddCfgEdge(cfg, body_exit, header);
      return header;
    }
    return entry;
  }

  static void ComputeRpo(const std::vector<CfgVertex>& cfg, size_t entry, std::vector<size_t>& rpo,
                         std::vector<size_t>& rpo_pos) {
    const size_t n = cfg.size();
    std::vector<bool> seen(n, false);
    std::vector<size_t> post_order;
    post_order.reserve(n);
    std::vector<std::pair<size_t, size_t>> stack;
    seen[entry] = true;
    stack.emplace_back(entry, 0);
    while (!stack.empty()) {
      auto& [u, idx] = stack.back();
      if (idx < cfg[u].succs.size()) {
        const size_t v = cfg[u].succs[idx++];
        if (!seen[v]) {
          seen[v] = true;
          stack.emplace_back(v, 0);
        }
      } else {
        post_order.push_back(u);
        stack.pop_back();
      }
    }
    rpo.assign(post_order.rbegin(), post_order.rend());
    rpo_pos.assign(n, n);
    for (size_t i = 0; i < rpo.size(); ++i) rpo_pos[rpo[i]] = i;
  }

  LiveStorageState TransferLiveStorage(const CfgVertex& vertex, LiveStorageState state) const {
    if (vertex.kind == CfgVertex::Kind::Site) {
      const auto& site = sites[vertex.index];
      if (enable_refresh_kill_) {
        for (const auto& [space, domain] : site.refreshed_domains) state.KillDomain(space, domain);
      }
      state.AddEffects(site.effects);
    } else if (vertex.kind == CfgVertex::Kind::Child) {
      state.AddEffects(children[vertex.index].effects);
    }
    return state;
  }

  void SolveLiveStorage(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                        const std::vector<size_t>& rpo_pos, size_t entry, std::vector<LiveStorageState>& live_in,
                        std::vector<LiveStorageState>& live_out) {
    const size_t n = cfg.size();
    live_in.assign(n, LiveStorageState{});
    live_out.assign(n, LiveStorageState{});
    std::vector<bool> visited(n, false);
    std::deque<size_t> queue;
    std::vector<bool> in_queue(n, false);

    auto eval = [&](size_t u) {
      LiveStorageState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(live_out[p]);
      }
      if (visited[u] && !(in_u != live_in[u])) return;
      live_in[u] = in_u;
      const LiveStorageState out_u = TransferLiveStorage(cfg[u], in_u);
      ++cost.dataflow_visits;
      const bool out_changed = !visited[u] || (out_u != live_out[u]);
      visited[u] = true;
      live_out[u] = out_u;
      if (out_changed) {
        for (const size_t v : cfg[u].succs) {
          if (!in_queue[v]) {
            queue.push_back(v);
            in_queue[v] = true;
          }
        }
      }
    };

    for (const size_t u : rpo) {
      (void)entry;
      eval(u);
    }
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u] && !in_queue[v]) {
          queue.push_back(v);
          in_queue[v] = true;
        }
      }
    }
    while (!queue.empty()) {
      const size_t u = queue.front();
      queue.pop_front();
      in_queue[u] = false;
      eval(u);
    }
  }

  void ExtractSourcePipeServices(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                                 const std::vector<LiveStorageState>& live_in,
                                 const std::vector<LiveStorageState>& live_out) {
    ConservativeAccessSet tracked_facts;
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      if (v.kind == CfgVertex::Kind::BranchMerge && !v.preds.empty()) {
        cost.control_snapshot_facts += live_out[v.preds[0]].FactCount();
        for (size_t i = 1; i < v.preds.size(); ++i) {
          cost.control_merge_facts += live_out[v.preds[i]].FactCount();
        }
        continue;
      }
      if (v.kind == CfgVertex::Kind::Child) {
        auto& entry = children[v.index];
        for (const auto& fact : entry.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          entry.incoming = live_in[u].ToAccessSet();
          cost.incoming_diagnostic_facts += entry.incoming.size();
        }
        std::set<std::pair<PipeType, PipeType>> services;
        for (const auto& effect : entry.effects) {
          for (const auto source : live_in[u].QueryConflicts(effect, cost)) {
            services.emplace(source, std::get<1>(effect));
          }
        }
        if (entry.eventful) {
          for (const auto source : entry.eventful->Pipes()) {
            for (const auto destination : entry.eventful->Pipes()) {
              if (source != destination) services.emplace(source, destination);
            }
          }
        }
        entry.services.assign(services.begin(), services.end());
        entry.required_services = entry.services;
        cost.baseline_services += services.size();
        cost.services += services.size();
        continue;
      }
      if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        for (const auto& fact : site.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          site.incoming = live_in[u].ToAccessSet();
          cost.incoming_diagnostic_facts += site.incoming.size();
        }
        if (enable_refresh_kill_) {
          for (const auto& [space, domain] : site.refreshed_domains) {
            cost.killed_facts += live_in[u].CountDomain(space, domain);
          }
        }
        std::set<PipeType> sources;
        for (const auto& effect : site.effects) {
          const auto conflicting = live_in[u].QueryConflicts(effect, cost);
          sources.insert(conflicting.begin(), conflicting.end());
        }
        site.required_source_pipes.assign(sources.begin(), sources.end());
        cost.baseline_services += sources.size();
      }
    }
  }

  DebtState TransferDebt(const CfgVertex& vertex, DebtState state, bool use_selected) const {
    if (vertex.kind == CfgVertex::Kind::Child) {
      const auto& entry = children[vertex.index];
      for (const auto& [src, dst] : entry.services) state.DeliverService(src, dst);
      state.ApplySummary(entry.completion);
      state.AddEffects(entry.effects, &entry.completion);
    } else if (vertex.kind == CfgVertex::Kind::Site) {
      const auto& site = sites[vertex.index];
      const auto destination = site.operation.Implementation().Pipe();
      const auto& pipes = use_selected ? site.source_pipes : site.required_source_pipes;
      for (const auto source : pipes) state.DeliverService(source, destination);
      if (enable_refresh_kill_) {
        for (const auto& [space, domain] : site.refreshed_domains) state.KillDomain(space, domain);
      }
      state.AddEffects(site.effects);
    }
    return state;
  }

  void SolveDebtFixedPoint(bool use_selected, const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                           const std::vector<size_t>& rpo_pos, std::vector<DebtState>& debt_in,
                           std::vector<DebtState>& debt_out) const {
    const size_t n = cfg.size();
    debt_in.assign(n, DebtState{});
    debt_out.assign(n, DebtState{});
    std::vector<bool> visited(n, false);
    std::deque<size_t> queue;
    std::vector<bool> in_queue(n, false);

    auto eval = [&](size_t u) {
      DebtState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(debt_out[p]);
      }
      if (visited[u] && !(in_u != debt_in[u])) return;
      debt_in[u] = in_u;
      const DebtState out_u = TransferDebt(cfg[u], std::move(in_u), use_selected);
      const bool out_changed = !visited[u] || (out_u != debt_out[u]);
      visited[u] = true;
      debt_out[u] = out_u;
      if (out_changed) {
        for (const size_t v : cfg[u].succs) {
          if (!in_queue[v]) {
            queue.push_back(v);
            in_queue[v] = true;
          }
        }
      }
    };

    for (const size_t u : rpo) eval(u);
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u] && !in_queue[v]) {
          queue.push_back(v);
          in_queue[v] = true;
        }
      }
    }
    while (!queue.empty()) {
      const size_t u = queue.front();
      queue.pop_front();
      in_queue[u] = false;
      eval(u);
    }
  }

  void SelectAndCertifyDebts(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                             const std::vector<size_t>& rpo_pos, size_t entry) {
    (void)entry;
    std::vector<DebtState> base_in, base_out;
    SolveDebtFixedPoint(/*use_selected=*/false, cfg, rpo, rpo_pos, base_in, base_out);

    // Greedy forward RPO selection, pulling back-edge baselines so loop-carried
    // demands are visible at loop headers prior to certification.
    std::vector<DebtState> rpo_out(cfg.size());
    std::vector<bool> rpo_done(cfg.size(), false);
    for (const size_t u : rpo) {
      DebtState cur_in;
      for (const size_t p : cfg[u].preds) {
        cur_in.MergeFrom(rpo_done[p] ? rpo_out[p] : base_out[p]);
      }
      if (cfg[u].kind == CfgVertex::Kind::Site) {
        auto& site = sites[cfg[u].index];
        const auto destination = site.operation.Implementation().Pipe();
        site.source_pipes = cur_in.Select(site.required_source_pipes, destination, site.effects);
      }
      rpo_out[u] = TransferDebt(cfg[u], std::move(cur_in), /*use_selected=*/true);
      rpo_done[u] = true;
    }

    // Closed-loop certification + monotone restoration until full CFG fixed point holds.
    for (;;) {
      std::vector<DebtState> sel_in, sel_out;
      SolveDebtFixedPoint(/*use_selected=*/true, cfg, rpo, rpo_pos, sel_in, sel_out);
      bool restored = false;
      for (const size_t u : rpo) {
        if (cfg[u].kind != CfgVertex::Kind::Site) continue;
        auto& site = sites[cfg[u].index];
        const auto destination = site.operation.Implementation().Pipe();
        DebtState after_services = sel_in[u];
        for (const auto source : site.source_pipes) after_services.DeliverService(source, destination);
        std::vector<PipeType> missing;
        for (const auto req : site.required_source_pipes) {
          if (!after_services.CoversConflicts(req, destination, site.effects)) missing.push_back(req);
        }
        if (!missing.empty()) {
          restored = true;
          std::set<PipeType> unioned(site.source_pipes.begin(), site.source_pipes.end());
          unioned.insert(missing.begin(), missing.end());
          site.source_pipes.clear();
          for (const auto req : site.required_source_pipes) {
            if (unioned.count(req)) site.source_pipes.push_back(req);
          }
        }
      }
      if (!restored) {
        for (const size_t u : rpo) {
          ++cost.certificate_visits;
          if (cfg[u].kind == CfgVertex::Kind::Child) {
            const auto& entry_child = children[cfg[u].index];
            INTERNAL_CHECK(entry_child.services == entry_child.required_services)
                << "Internal error: protected child/rearming services changed";
          } else if (cfg[u].kind == CfgVertex::Kind::Site) {
            const auto& selected = sites[cfg[u].index];
            size_t next = 0;
            const auto destination = selected.operation.Implementation().Pipe();
            DebtState known = sel_in[u];
            for (const auto source : selected.required_source_pipes) {
              if (next < selected.source_pipes.size() && selected.source_pipes[next] == source) {
                known.DeliverService(source, destination);
                ++next;
              }
            }
            INTERNAL_CHECK(next == selected.source_pipes.size())
                << "Internal error: conservative service selection added or reordered a service";
            for (const auto source : selected.required_source_pipes) {
              INTERNAL_CHECK(known.CoversConflicts(source, destination, selected.effects))
                  << "Internal error: conservative service selection omits a payload prerequisite";
            }
            const size_t omitted = selected.required_source_pipes.size() - selected.source_pipes.size();
            cost.certificate_omissions += omitted;
            cost.suppressed_services += omitted;
            cost.services += selected.source_pipes.size();
          }
        }
        INTERNAL_CHECK(cost.certificate_omissions == cost.suppressed_services)
            << "Internal error: conservative service certificate omitted a proposed deletion";
        break;
      }
    }
  }

  void RefineStorage() {
    std::vector<std::vector<RawEffect>> raw(sites.size());
    for (size_t site = 0; site < sites.size(); ++site) {
      const auto& operation = sites[site].operation;
      const auto& call = operation.Operation();
      const auto pipe = operation.Implementation().Pipe();
      const auto append = [&](MemorySpace space, BufferAccess mode,
                              std::optional<backend::BufferSyncByteWindow> enclosure, ExprPtr root,
                              bool candidate_full_overwrite,
                              std::optional<backend::BufferSyncByteWindow> exact_full_window) {
        if (mode == BufferAccess::Read || mode == BufferAccess::ReadWrite)
          raw[site].push_back({space, pipe, false, enclosure, root, false, std::nullopt});
        if (mode == BufferAccess::Write || mode == BufferAccess::ReadWrite)
          raw[site].push_back(
              {space, pipe, true, enclosure, root, candidate_full_overwrite, exact_full_window});
      };
      for (size_t index = 0; index < operation.ArgumentEffects().size(); ++index) {
        const auto mode = operation.ArgumentEffects()[index].data;
        if (mode == BufferAccess::None) continue;
        const auto& operand = call->args_[index];
        const auto space = Space(operand);
        std::optional<backend::BufferSyncByteWindow> enclosure;
        std::optional<backend::BufferSyncByteWindow> exact_full_window;
        ExprPtr root;
        bool candidate_full_overwrite = false;
        if (space == MemorySpace::DDR) {
          if (disjoint_gm_) {
            if (const auto found = gm_roots_.find(operand.get()); found != gm_roots_.end())
              root = found->second;
          }
        } else {
          ++cost.enclosure_lookups;
          enclosure = placement_.Window(operand);
          if (enable_refresh_kill_ && IsFullOverwriteOperand(call, mode, space, operand)) {
            exact_full_window = placement_.ExactFullWindow(operand);
            candidate_full_overwrite = exact_full_window.has_value();
          }
        }
        append(space, mode, enclosure, std::move(root), candidate_full_overwrite, exact_full_window);
      }
      for (const auto& effect : operation.Implementation().ImplicitEffects()) {
        std::optional<backend::BufferSyncByteWindow> enclosure;
        if (effect.space != MemorySpace::DDR && effect.begin >= 0 && effect.end > effect.begin)
          enclosure = backend::BufferSyncByteWindow{effect.begin, effect.end};
        append(effect.space, effect.data, enclosure, nullptr, false, std::nullopt);
      }
    }
    StorageClasses classes;
    classes.Build(raw);
    for (size_t site = 0; site < sites.size(); ++site) {
      auto& effects = sites[site].effects;
      effects.clear();
      std::set<std::pair<MemorySpace, size_t>> refreshed;
      for (const auto& effect : raw[site]) {
        const size_t domain = classes.Domain(effect);
        effects.emplace(effect.space, effect.pipe, effect.write, domain);
        if (effect.candidate_full_overwrite && effect.exact_full_window && domain > 0) {
          const auto& dom = classes.domains[domain - 1];
          if (dom.enclosure && dom.enclosure->begin == effect.exact_full_window->begin &&
              dom.enclosure->end == effect.exact_full_window->end) {
            refreshed.emplace(effect.space, domain);
          }
        }
      }
      sites[site].refreshed_domains.assign(refreshed.begin(), refreshed.end());
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
    sites.push_back({cut, std::move(operation), std::move(effects), {}, {}, {}, {}});
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
  bool enable_refresh_kill_;
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
    bool retain_incoming_diagnostics, bool automatic_children, bool enable_refresh_kill) {
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
  Builder builder(function, target, children, selected, retain_incoming_diagnostics, enable_refresh_kill);
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
                                  retain_incoming_diagnostics, builder.disjoint_gm_, enable_refresh_kill,
                                  std::move(selection));
}
}  // namespace pypto::ir::sync
