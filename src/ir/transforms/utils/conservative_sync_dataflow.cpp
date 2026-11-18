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
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pypto/backend/common/buffer_conversion_recipes.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/core/any_cast.h"
#include "pypto/core/logging.h"
#include "pypto/ir/kind_traits.h"
#include "pypto/ir/memref.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/sync.h"
#include "pypto/ir/transforms/utils/buffer_runtime_valid_analysis.h"
#include "pypto/ir/type.h"
#include "src/ir/transforms/utils/native_sync_placement.h"
#include "src/ir/transforms/utils/sync_scalar_readiness.h"

namespace pypto::ir::sync {
namespace {
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
  if (name == "buffer.ci_workspace") {
    return mode == BufferAccess::Write && !call->args_.empty() && operand == call->args_.back();
  }
  if (name == "buffer.matmul_acc") {
    return mode == BufferAccess::ReadWrite;
  }
  if (backend::FindBufferElementwiseRecipe(name)) {
    return mode == BufferAccess::Write && !call->args_.empty() && operand == call->args_.back();
  }
  if (backend::FindBufferConversionRecipe(name)) {
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

  // Batched 1-way relay from all sources in `sources_mask` to `dst` in one pass.
  void DeliverSources(PipeMask sources_mask, PipeType dst) {
    const PipeMask dst_mask = PipeToMask(dst);
    if (sources_mask == 0 || dst_mask == 0) return;
    for (auto it = pending_.begin(); it != pending_.end();) {
      const PipeMask ready = (~it->second & kAllPhysicalPipesMask) | PipeToMask(it->first.src);
      if ((ready & sources_mask) != 0) it->second &= ~dst_mask;
      if (it->second == 0) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
  }

  // Apply a precomputed sequential relay matrix in one pass over `pending_`.
  void ApplyServiceMatrix(const std::array<PipeMask, kPhysicalPipeCount>& matrix) {
    for (auto it = pending_.begin(); it != pending_.end();) {
      const PipeMask ready = (~it->second & kAllPhysicalPipesMask) | PipeToMask(it->first.src);
      PipeMask cleared = 0;
      for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
        if ((ready & (PipeMask{1} << p)) != 0) cleared |= matrix[p];
      }
      it->second &= ~cleared;
      if (it->second == 0) {
        it = pending_.erase(it);
      } else {
        ++it;
      }
    }
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
        pending_[{space, domain, write, pipe}] |= initial;
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
      children.push_back({child->Scope(), {}, {}, {}, {}, {}, CompletionSummary::Identity()});
    }
    for (const auto& child : eventful) {
      CHECK(child) << "Conservative synchronization: missing eventful child certificate";
      child->ValidateFor(function, target_);
      CHECK(child_index_.emplace(child->Scope().get(), children.size()).second)
          << "Conservative synchronization: duplicate retained child scope";
      children.push_back({child->Scope(), {}, {}, {}, child, {}, CompletionSummary::Identity()});
    }
  }

  CompletionSummary ComputeEventfulCompletionSummary(const ConservativeSyncChildEntry& entry) const {
    CompletionSummary base = CompletionSummary::Identity();
    base.has_bypass_path = true;
    for (const auto& fact : entry.effects) {
      const PipeType pipe = std::get<1>(fact);
      if (pipe != PipeType::ALL) base.active |= PipeToMask(pipe);
    }
    if (!entry.eventful || !entry.eventful->LogicalRecipe()) return base;

    std::vector<CompletionVertex> vertices;
    std::vector<std::pair<size_t, size_t>> edges;
    const size_t entry_v = vertices.size();
    vertices.push_back(CompletionVertex{});
    std::unordered_map<const Expr*, PipeMask> local_tokens;
    PipeMask written_pipes = 0;

    auto find_pipe_arg = [](const CallPtr& call, const char* key) -> std::optional<PipeType> {
      for (const auto& [name, val] : call->kwargs_) {
        if (name != key) continue;
        return static_cast<PipeType>(AnyCast<int>(val, name));
      }
      for (const auto& [name, val] : call->attrs_) {
        if (name != key) continue;
        return static_cast<PipeType>(AnyCast<int>(val, name));
      }
      return std::nullopt;
    };

    auto new_vertex = [&](CompletionVertex vertex) {
      const size_t id = vertices.size();
      vertices.push_back(std::move(vertex));
      return id;
    };

    auto emit_payload = [&](const CallPtr& call, size_t cur) -> size_t {
      if (!call || GetLogicalSyncKind(call->op_) != LogicalSyncKind::None ||
          backend::QualifyBufferSyncNonPayload(call, target_))
        return cur;
      const auto op = backend::QualifyBufferSyncOperation(call, function_->func_type_, target_);
      const PipeMask mask = PipeToMask(op.Implementation().Pipe());
      if (mask == 0) return cur;
      written_pipes |= mask;
      for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~mask;
      CompletionVertex op_v;
      op_v.write_pipes = mask;
      const size_t v = new_vertex(std::move(op_v));
      edges.emplace_back(cur, v);
      return v;
    };

    std::function<size_t(const StmtPtr&, size_t)> build = [&](const StmtPtr& stmt, size_t cur) -> size_t {
      if (!stmt) return cur;
      if (const auto seq = As<SeqStmts>(stmt)) {
        for (const auto& child : seq->stmts_) cur = build(child, cur);
        return cur;
      }
      if (const auto branch = As<IfStmt>(stmt)) {
        const auto saved_tokens = local_tokens;
        const PipeMask saved_written = written_pipes;
        written_pipes = 0;
        const size_t then_exit = build(branch->then_body_, cur);
        const auto then_tokens = local_tokens;
        const PipeMask then_written = written_pipes;
        local_tokens = saved_tokens;
        written_pipes = 0;
        const size_t else_exit = branch->else_body_ ? build(*branch->else_body_, cur) : cur;
        const PipeMask else_written = written_pipes;
        written_pipes = saved_written | then_written | else_written;
        for (auto it = local_tokens.begin(); it != local_tokens.end();) {
          const auto found = then_tokens.find(it->first);
          if (found == then_tokens.end()) {
            it = local_tokens.erase(it);
          } else {
            it->second &= found->second;
            ++it;
          }
        }
        const size_t merge = new_vertex(CompletionVertex{});
        edges.emplace_back(then_exit, merge);
        edges.emplace_back(else_exit, merge);
        return merge;
      }
      if (const auto loop = As<ForStmt>(stmt)) {
        const auto saved_tokens = local_tokens;
        const PipeMask outer_written = written_pipes;
        written_pipes = 0;
        const size_t hdr = new_vertex(CompletionVertex{});
        edges.emplace_back(cur, hdr);
        local_tokens.clear();
        const size_t body_exit = build(loop->body_, hdr);
        edges.emplace_back(body_exit, hdr);
        const PipeMask loop_written = written_pipes;
        local_tokens = saved_tokens;
        for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~loop_written;
        written_pipes = outer_written | loop_written;
        return hdr;
      }
      if (const auto loop = As<WhileStmt>(stmt)) {
        const auto saved_tokens = local_tokens;
        const PipeMask outer_written = written_pipes;
        written_pipes = 0;
        const size_t hdr = new_vertex(CompletionVertex{});
        edges.emplace_back(cur, hdr);
        local_tokens.clear();
        const size_t body_exit = build(loop->body_, hdr);
        edges.emplace_back(body_exit, hdr);
        const PipeMask loop_written = written_pipes;
        local_tokens = saved_tokens;
        for (auto& [tok, tok_mask] : local_tokens) tok_mask &= ~loop_written;
        written_pipes = outer_written | loop_written;
        return hdr;
      }
      if (const auto assign = As<AssignStmt>(stmt)) {
        if (As<SyncTokenType>(assign->var_->GetType())) {
          if (const auto rhs = AsVarLike(assign->value_)) {
            const auto found = local_tokens.find(rhs.get());
            local_tokens[assign->var_.get()] = found == local_tokens.end() ? 0 : found->second;
          } else if (const auto call = As<Call>(assign->value_)) {
            if (IsOp(call, "sync.record")) {
              if (const auto pipe = find_pipe_arg(call, "pipe")) {
                local_tokens[assign->var_.get()] = PipeToMask(*pipe);
              }
            } else if (IsOp(call, "sync.completed")) {
              local_tokens[assign->var_.get()] = 0;
            } else if (IsOp(call, "sync.join")) {
              PipeMask mask = 0;
              bool all_known = !call->args_.empty();
              for (const auto& arg : call->args_) {
                const auto v = AsVarLike(arg);
                const auto found = v ? local_tokens.find(v.get()) : local_tokens.end();
                if (found == local_tokens.end()) {
                  all_known = false;
                  break;
                }
                mask |= found->second;
              }
              local_tokens[assign->var_.get()] = all_known ? mask : 0;
            } else {
              local_tokens.erase(assign->var_.get());
            }
          } else {
            local_tokens.erase(assign->var_.get());
          }
          return cur;
        }
        return emit_payload(As<Call>(assign->value_), cur);
      }
      if (const auto eval = As<EvalStmt>(stmt)) {
        if (const auto call = As<Call>(eval->expr_)) {
          if (IsOp(call, "sync.wait") && !call->args_.empty()) {
            const auto v = AsVarLike(call->args_[0]);
            if (const auto found = v ? local_tokens.find(v.get()) : local_tokens.end();
                found != local_tokens.end() && found->second != 0) {
              if (const auto dst = find_pipe_arg(call, "dst_pipe")) {
                CompletionVertex sync_v;
                for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
                  if ((found->second & (PipeMask{1} << p)) != 0) {
                    sync_v.pre_syncs.push_back(
                        CompletionSyncEdge{static_cast<PipeType>(p), *dst, /*bidirectional=*/false});
                  }
                }
                const size_t node = new_vertex(std::move(sync_v));
                edges.emplace_back(cur, node);
                return node;
              }
            }
            return cur;
          }
          return emit_payload(call, cur);
        }
      }
      return cur;
    };

    const size_t tail = build(entry.eventful->LogicalRecipe(), entry_v);
    const size_t exit_v = new_vertex(CompletionVertex{});
    edges.emplace_back(tail, exit_v);
    auto summary = SolveCompletionTransfer(vertices, edges, entry_v, exit_v).summary;
    const PipeMask added_active = base.active & ~summary.active;
    for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
      if ((added_active & (PipeMask{1} << p)) != 0) {
        summary.post_write_safe[p] = PipeMask{1} << p;
      }
    }
    summary.active |= base.active;
    return summary;
  }

  void Build() {
    Read(function_->body_);
    RefineStorage();
    for (auto& child : children) {
      CHECK(seen_children_.count(child.scope.get()) != 0) << "Retained child is outside the original semantic input";
      CollectChildEffects(child.scope, child.effects);
      child.completion = ComputeEventfulCompletionSummary(child);
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

    bool has_backedge = false;
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) {
          has_backedge = true;
          break;
        }
      }
      if (has_backedge) break;
    }

    if (!has_backedge) {
      return SolveAcyclic(cfg, rpo, cfg_exit);
    }

    std::vector<LiveStorageState> live_in, live_out;
    SolveLiveStorage(cfg, rpo, rpo_pos, cfg_entry, live_in, live_out);
    ExtractSourcePipeServices(cfg, rpo, live_in, live_out);
    SelectAndCertifyDebts(cfg, rpo, rpo_pos, cfg_entry);

    return retain_incoming_diagnostics_ ? live_out[cfg_exit].ToAccessSet() : ConservativeAccessSet{};
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
      if (IsSpmdAtMostOneTripFor(loop, ssa_defs_)) {
        const size_t body_exit = BuildSubCfg(loop->body_, entry, cfg);
        const size_t merge = cfg.size();
        cfg.push_back({CfgVertex::Kind::BranchMerge, 0, {}, {}});
        AddCfgEdge(cfg, entry, merge);
        AddCfgEdge(cfg, body_exit, merge);
        return merge;
      }
      const size_t header = cfg.size();
      cfg.push_back({CfgVertex::Kind::Nop, 0, {}, {}});
      AddCfgEdge(cfg, entry, header);
      const size_t body_exit = BuildSubCfg(loop->body_, header, cfg);
      AddCfgEdge(cfg, body_exit, header);
      const auto start = As<ConstInt>(loop->start_);
      const auto stop = As<ConstInt>(loop->stop_);
      const auto step = As<ConstInt>(loop->step_);
      if (start && stop && step && step->value_ > 0 && stop->value_ > start->value_) {
        return body_exit;
      }
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

  ConservativeAccessSet SolveAcyclic(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                                     size_t cfg_exit) {
    const size_t n = cfg.size();
    std::vector<bool> has_reaching_child(n, false);
    {
      std::queue<size_t> q;
      for (size_t w = 0; w < n; ++w) {
        if (cfg[w].kind != CfgVertex::Kind::Child) continue;
        for (const size_t nxt : cfg[w].succs) {
          if (!has_reaching_child[nxt]) {
            has_reaching_child[nxt] = true;
            q.push(nxt);
          }
        }
      }
      while (!q.empty()) {
        const size_t x = q.front();
        q.pop();
        for (const size_t nxt : cfg[x].succs) {
          if (!has_reaching_child[nxt]) {
            has_reaching_child[nxt] = true;
            q.push(nxt);
          }
        }
      }
    }

    // Pass 1: Fused live-storage transfer and conflict extraction with immediate
    // predecessor state release so straight-line sequences hold O(1) active maps.
    std::vector<LiveStorageState> live_out(n);
    std::vector<size_t> rem_succs(n);
    for (size_t u = 0; u < n; ++u) rem_succs[u] = cfg[u].succs.size();
    ConservativeAccessSet tracked_facts;
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      if (v.kind == CfgVertex::Kind::BranchMerge && !v.preds.empty()) {
        cost.control_snapshot_facts += live_out[v.preds[0]].FactCount();
        for (size_t i = 1; i < v.preds.size(); ++i) {
          cost.control_merge_facts += live_out[v.preds[i]].FactCount();
        }
      }
      LiveStorageState in_u;
      if (v.preds.size() == 1) {
        const size_t p = v.preds[0];
        if (--rem_succs[p] == 0) {
          in_u = std::move(live_out[p]);
        } else {
          in_u = live_out[p];
        }
      } else {
        for (const size_t p : v.preds) {
          in_u.MergeFrom(live_out[p]);
          if (--rem_succs[p] == 0) live_out[p] = LiveStorageState{};
        }
      }
      if (v.kind == CfgVertex::Kind::Child) {
        auto& entry = children[v.index];
        for (const auto& fact : entry.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          entry.incoming = in_u.ToAccessSet();
          cost.incoming_diagnostic_facts += entry.incoming.size();
        }
        std::set<std::pair<PipeType, PipeType>> services;
        for (const auto& effect : entry.effects) {
          for (const auto source : in_u.QueryConflicts(effect, cost)) {
            services.emplace(source, std::get<1>(effect));
          }
        }
        entry.first_visit = !has_reaching_child[u];
        if (has_reaching_child[u] && entry.eventful) {
          for (const auto& dir : entry.eventful->RearmDirections()) services.insert(dir);
        }
        entry.services.assign(services.begin(), services.end());
        entry.required_services = entry.services;
        cost.baseline_services += services.size();
        cost.services += services.size();
        std::array<PipeMask, kPhysicalPipeCount> matrix{};
        for (const auto& [src, dst] : entry.services) {
          const PipeMask sm = PipeToMask(src);
          const PipeMask dm = PipeToMask(dst);
          if (sm == 0 || dm == 0) continue;
          for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
            const PipeMask p_bit = PipeMask{1} << p;
            if (((p_bit | matrix[p]) & sm) != 0) matrix[p] |= dm;
          }
        }
        if (child_service_matrices_.size() <= v.index) child_service_matrices_.resize(v.index + 1);
        child_service_matrices_[v.index] = matrix;
      } else if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        for (const auto& fact : site.effects) {
          if (tracked_facts.insert(fact).second) ++cost.conflict_summary_facts;
        }
        if (retain_incoming_diagnostics_) {
          site.incoming = in_u.ToAccessSet();
          cost.incoming_diagnostic_facts += site.incoming.size();
        }
        if (enable_refresh_kill_) {
          for (const auto& [space, domain] : site.refreshed_domains) {
            cost.killed_facts += in_u.CountDomain(space, domain);
          }
        }
        std::set<PipeType> sources;
        for (const auto& effect : site.effects) {
          const auto conflicting = in_u.QueryConflicts(effect, cost);
          sources.insert(conflicting.begin(), conflicting.end());
        }
        site.required_source_pipes.assign(sources.begin(), sources.end());
        cost.baseline_services += sources.size();
      }
      ++cost.dataflow_visits;
      live_out[u] = TransferLiveStorage(v, std::move(in_u));
      if (u != cfg_exit && rem_succs[u] == 0) live_out[u] = LiveStorageState{};
    }
    ConservativeAccessSet exit_set =
        retain_incoming_diagnostics_ ? live_out[cfg_exit].ToAccessSet() : ConservativeAccessSet{};
    live_out.clear();
    live_out.shrink_to_fit();

    // Pass 2: Fused debt selection and certification on the DAG with immediate
    // predecessor state release.
    std::vector<DebtState> debt_out(n);
    for (size_t u = 0; u < n; ++u) rem_succs[u] = cfg[u].succs.size();
    for (const size_t u : rpo) {
      const auto& v = cfg[u];
      DebtState cur_in;
      if (v.preds.size() == 1) {
        const size_t p = v.preds[0];
        if (--rem_succs[p] == 0) {
          cur_in = std::move(debt_out[p]);
        } else {
          cur_in = debt_out[p];
        }
      } else {
        for (const size_t p : v.preds) {
          cur_in.MergeFrom(debt_out[p]);
          if (--rem_succs[p] == 0) debt_out[p] = DebtState{};
        }
      }
      ++cost.certificate_visits;
      if (v.kind == CfgVertex::Kind::Child) {
        const auto& entry_child = children[v.index];
        INTERNAL_CHECK(entry_child.services == entry_child.required_services)
            << "Internal error: protected child/rearming services changed";
        debt_out[u] = TransferDebt(v, std::move(cur_in), /*use_selected=*/true);
      } else if (v.kind == CfgVertex::Kind::Site) {
        auto& site = sites[v.index];
        const auto destination = site.operation.Implementation().Pipe();
        site.source_pipes = cur_in.Select(site.required_source_pipes, destination, site.effects);
        PipeMask sources_mask = 0;
        for (const auto source : site.source_pipes) sources_mask |= PipeToMask(source);
        cur_in.DeliverSources(sources_mask, destination);
        std::vector<PipeType> missing;
        for (const auto req : site.required_source_pipes) {
          if (!cur_in.CoversConflicts(req, destination, site.effects)) missing.push_back(req);
        }
        if (!missing.empty()) {
          std::set<PipeType> unioned(site.source_pipes.begin(), site.source_pipes.end());
          unioned.insert(missing.begin(), missing.end());
          site.source_pipes.clear();
          PipeMask extra_mask = 0;
          for (const auto req : site.required_source_pipes) {
            if (unioned.count(req)) {
              site.source_pipes.push_back(req);
              extra_mask |= PipeToMask(req);
            }
          }
          cur_in.DeliverSources(extra_mask, destination);
        }
        size_t next = 0;
        for (const auto source : site.required_source_pipes) {
          if (next < site.source_pipes.size() && site.source_pipes[next] == source) ++next;
        }
        INTERNAL_CHECK(next == site.source_pipes.size())
            << "Internal error: conservative service selection added or reordered a service";
        for (const auto source : site.required_source_pipes) {
          INTERNAL_CHECK(cur_in.CoversConflicts(source, destination, site.effects))
              << "Internal error: conservative service selection omits a payload prerequisite";
        }
        const size_t omitted = site.required_source_pipes.size() - site.source_pipes.size();
        cost.certificate_omissions += omitted;
        cost.suppressed_services += omitted;
        cost.services += site.source_pipes.size();
        if (enable_refresh_kill_) {
          for (const auto& [space, domain] : site.refreshed_domains) cur_in.KillDomain(space, domain);
        }
        cur_in.AddEffects(site.effects);
        debt_out[u] = std::move(cur_in);
      } else {
        debt_out[u] = std::move(cur_in);
      }
      if (rem_succs[u] == 0) debt_out[u] = DebtState{};
    }
    INTERNAL_CHECK(cost.certificate_omissions == cost.suppressed_services)
        << "Internal error: conservative service certificate omitted a proposed deletion";
    return exit_set;
  }

  void SolveLiveStorage(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                        const std::vector<size_t>& rpo_pos, size_t entry, std::vector<LiveStorageState>& live_in,
                        std::vector<LiveStorageState>& live_out) {
    const size_t n = cfg.size();
    live_in.assign(n, LiveStorageState{});
    live_out.assign(n, LiveStorageState{});
    std::vector<bool> visited(n, false);
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>,
                        std::greater<std::pair<size_t, size_t>>>
        worklist;
    std::vector<bool> in_queue(n, false);
    auto push_vertex = [&](size_t v) {
      if (!in_queue[v]) {
        worklist.emplace(rpo_pos[v], v);
        in_queue[v] = true;
      }
    };

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
        for (const size_t v : cfg[u].succs) push_vertex(v);
      }
    };

    for (const size_t u : rpo) {
      (void)entry;
      eval(u);
    }
    for (const size_t u : rpo) {
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) push_vertex(v);
      }
    }
    while (!worklist.empty()) {
      const size_t u = worklist.top().second;
      worklist.pop();
      in_queue[u] = false;
      eval(u);
    }
  }

  void ExtractSourcePipeServices(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                                 const std::vector<LiveStorageState>& live_in,
                                 const std::vector<LiveStorageState>& live_out) {
    std::vector<bool> has_reaching_child(cfg.size(), false);
    {
      std::queue<size_t> q;
      for (size_t w = 0; w < cfg.size(); ++w) {
        if (cfg[w].kind != CfgVertex::Kind::Child) continue;
        for (const size_t nxt : cfg[w].succs) {
          if (!has_reaching_child[nxt]) {
            has_reaching_child[nxt] = true;
            q.push(nxt);
          }
        }
      }
      while (!q.empty()) {
        const size_t x = q.front();
        q.pop();
        for (const size_t nxt : cfg[x].succs) {
          if (!has_reaching_child[nxt]) {
            has_reaching_child[nxt] = true;
            q.push(nxt);
          }
        }
      }
    }
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
        entry.first_visit = !has_reaching_child[u];
        if (has_reaching_child[u] && entry.eventful) {
          for (const auto& dir : entry.eventful->RearmDirections()) services.insert(dir);
        }
        entry.services.assign(services.begin(), services.end());
        entry.required_services = entry.services;
        cost.baseline_services += services.size();
        cost.services += services.size();
        std::array<PipeMask, kPhysicalPipeCount> matrix{};
        for (const auto& [src, dst] : entry.services) {
          const PipeMask sm = PipeToMask(src);
          const PipeMask dm = PipeToMask(dst);
          if (sm == 0 || dm == 0) continue;
          for (size_t p = 0; p < kPhysicalPipeCount; ++p) {
            const PipeMask p_bit = PipeMask{1} << p;
            if (((p_bit | matrix[p]) & sm) != 0) matrix[p] |= dm;
          }
        }
        if (child_service_matrices_.size() <= v.index) child_service_matrices_.resize(v.index + 1);
        child_service_matrices_[v.index] = matrix;
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
      state.ApplyServiceMatrix(child_service_matrices_[vertex.index]);
      state.ApplySummary(entry.completion);
      state.AddEffects(entry.effects, &entry.completion);
    } else if (vertex.kind == CfgVertex::Kind::Site) {
      const auto& site = sites[vertex.index];
      const auto destination = site.operation.Implementation().Pipe();
      const auto& pipes = use_selected ? site.source_pipes : site.required_source_pipes;
      PipeMask sources_mask = 0;
      for (const auto source : pipes) sources_mask |= PipeToMask(source);
      state.DeliverSources(sources_mask, destination);
      if (enable_refresh_kill_) {
        for (const auto& [space, domain] : site.refreshed_domains) state.KillDomain(space, domain);
      }
      state.AddEffects(site.effects);
    }
    return state;
  }

  void RunDebtWorklist(bool use_selected, const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo_pos,
                       std::vector<size_t> initial_dirty, std::vector<DebtState>& debt_in,
                       std::vector<DebtState>& debt_out, std::vector<bool>& visited) const {
    const size_t n = cfg.size();
    std::priority_queue<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>,
                        std::greater<std::pair<size_t, size_t>>>
        worklist;
    std::vector<bool> in_queue(n, false);
    for (const size_t u : initial_dirty) {
      if (!in_queue[u]) {
        worklist.emplace(rpo_pos[u], u);
        in_queue[u] = true;
      }
    }
    while (!worklist.empty()) {
      const size_t u = worklist.top().second;
      worklist.pop();
      in_queue[u] = false;
      DebtState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(debt_out[p]);
      }
      const DebtState out_u = TransferDebt(cfg[u], in_u, use_selected);
      const bool in_changed = !visited[u] || (in_u != debt_in[u]);
      const bool out_changed = !visited[u] || (out_u != debt_out[u]);
      if (!in_changed && !out_changed) continue;
      visited[u] = true;
      debt_in[u] = std::move(in_u);
      debt_out[u] = std::move(out_u);
      if (out_changed) {
        for (const size_t v : cfg[u].succs) {
          if (!in_queue[v]) {
            worklist.emplace(rpo_pos[v], v);
            in_queue[v] = true;
          }
        }
      }
    }
  }

  void SolveDebtFixedPoint(bool use_selected, const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                           const std::vector<size_t>& rpo_pos, std::vector<DebtState>& debt_in,
                           std::vector<DebtState>& debt_out, std::vector<bool>& visited) const {
    const size_t n = cfg.size();
    debt_in.assign(n, DebtState{});
    debt_out.assign(n, DebtState{});
    visited.assign(n, false);

    std::vector<size_t> back_targets;
    for (const size_t u : rpo) {
      DebtState in_u;
      for (const size_t p : cfg[u].preds) {
        if (visited[p]) in_u.MergeFrom(debt_out[p]);
      }
      debt_in[u] = in_u;
      const DebtState out_u = TransferDebt(cfg[u], std::move(in_u), use_selected);
      visited[u] = true;
      debt_out[u] = out_u;
      for (const size_t v : cfg[u].succs) {
        if (rpo_pos[v] <= rpo_pos[u]) back_targets.push_back(v);
      }
    }
    RunDebtWorklist(use_selected, cfg, rpo_pos, std::move(back_targets), debt_in, debt_out, visited);
  }

  void SelectAndCertifyDebts(const std::vector<CfgVertex>& cfg, const std::vector<size_t>& rpo,
                             const std::vector<size_t>& rpo_pos, size_t entry) {
    (void)entry;
    std::vector<DebtState> base_in, base_out;
    std::vector<bool> base_visited;
    SolveDebtFixedPoint(/*use_selected=*/false, cfg, rpo, rpo_pos, base_in, base_out, base_visited);

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

    // Closed-loop certification + incremental monotone restoration until full CFG fixed point holds.
    std::vector<DebtState> sel_in, sel_out;
    std::vector<bool> sel_visited;
    SolveDebtFixedPoint(/*use_selected=*/true, cfg, rpo, rpo_pos, sel_in, sel_out, sel_visited);
    std::vector<size_t> dirty = rpo;
    for (;;) {
      std::vector<size_t> changed_vertices;
      for (const size_t u : dirty) {
        if (cfg[u].kind != CfgVertex::Kind::Site) continue;
        auto& site = sites[cfg[u].index];
        const auto destination = site.operation.Implementation().Pipe();
        DebtState after_services = sel_in[u];
        PipeMask sources_mask = 0;
        for (const auto source : site.source_pipes) sources_mask |= PipeToMask(source);
        after_services.DeliverSources(sources_mask, destination);
        std::vector<PipeType> missing;
        for (const auto req : site.required_source_pipes) {
          if (!after_services.CoversConflicts(req, destination, site.effects)) missing.push_back(req);
        }
        if (!missing.empty()) {
          std::set<PipeType> unioned(site.source_pipes.begin(), site.source_pipes.end());
          unioned.insert(missing.begin(), missing.end());
          site.source_pipes.clear();
          for (const auto req : site.required_source_pipes) {
            if (unioned.count(req)) site.source_pipes.push_back(req);
          }
          changed_vertices.push_back(u);
        }
      }
      if (!changed_vertices.empty()) {
        RunDebtWorklist(/*use_selected=*/true, cfg, rpo_pos, changed_vertices, sel_in, sel_out, sel_visited);
        dirty = rpo;
        continue;
      }
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
          PipeMask sources_mask = 0;
          for (const auto source : selected.required_source_pipes) {
            if (next < selected.source_pipes.size() && selected.source_pipes[next] == source) {
              sources_mask |= PipeToMask(source);
              ++next;
            }
          }
          known.DeliverSources(sources_mask, destination);
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

  void CollectChildEffects(const StmtPtr& statement, ConservativeAccessSet& out) const {
    if (!statement) return;
    if (const auto site = site_index_.find(statement.get()); site != site_index_.end()) {
      Union(out, sites[site->second].effects);
    } else if (const auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) CollectChildEffects(child, out);
    } else if (const auto branch = As<IfStmt>(statement)) {
      CollectChildEffects(branch->then_body_, out);
      if (branch->else_body_) CollectChildEffects(*branch->else_body_, out);
    } else if (const auto loop = As<ForStmt>(statement)) {
      CollectChildEffects(loop->body_, out);
    } else if (const auto loop = As<WhileStmt>(statement)) {
      CollectChildEffects(loop->body_, out);
    }
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
    sites.push_back({cut, std::move(operation), std::move(effects), {}, {}, {}, {}});
    inside_child_.push_back(native_child_depth_ != 0);
    // The ordinary service protects the scalar instruction's storage read.
    // Only its selected blocking native return then makes the original SSA
    // available to following addresses/control; it is never a pure query.
    if (ready_result) Bind(result);
  }

  End Read(const StmtPtr& statement) {
    if (child_index_.count(statement.get())) seen_children_.insert(statement.get());
    ++cost.statements;
    if (const auto seq = As<SeqStmts>(statement)) {
      End end;
      for (const auto& child : seq->stmts_) {
        CHECK_SPAN(!end.Terminal(), child->span_)
            << "Conservative synchronization: statements after a terminal cut are unavailable";
        end = Read(child);
      }
      return end;
    }
    if (const auto assignment = As<AssignStmt>(statement)) {
      ssa_defs_[assignment->var_.get()] = assignment->value_;
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
      }
      --control_depth_;
      CHECK_SPAN(!then_end.returned && !else_end.returned, statement->span_)
          << "Conservative synchronization: nested invocation exits are unavailable";
      const auto count = branch->return_vars_.size();
      CHECK_SPAN(then_end.yield.value_or(0) == count && else_end.yield.value_or(0) == count, statement->span_)
          << "Conservative synchronization: branch result/yield arity mismatch";
      for (const auto& variable : branch->return_vars_) Bind(variable);
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
  }

  FunctionPtr function_;
  std::string target_;
  NativeSyncPlacement placement_;
  bool retain_incoming_diagnostics_;
  bool enable_refresh_kill_;
  std::unordered_map<const Expr*, ExprPtr> gm_roots_;
  std::unordered_map<const Var*, ExprPtr> ssa_defs_;
  std::vector<std::vector<const Var*>> scopes_;
  std::unordered_set<const Var*> ready_;
  size_t control_depth_ = 0;
  size_t native_child_depth_ = 0;
  std::vector<bool> inside_child_;
  std::unordered_map<const Stmt*, size_t> child_index_;
  std::unordered_set<const Stmt*> seen_children_;
  std::vector<std::array<PipeMask, kPhysicalPipeCount>> child_service_matrices_;
  std::unordered_map<const Stmt*, size_t> site_index_;
};
}  // namespace

ConservativeSyncAnalysis AnalyzeConservativeSync(
    const FunctionPtr& function, const std::string& target,
    const std::vector<std::shared_ptr<const NativeSyncChild>>& children,
    const std::vector<std::shared_ptr<const EventfulSyncChild>>& eventful_children,
    bool retain_incoming_diagnostics, bool automatic_children, bool enable_refresh_kill,
    const SyncParameterBounds& parameters, const std::optional<std::pair<int64_t, int64_t>>& blocks) {
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
    selection = PrepareAutomaticSyncChildren(function, target, parameters, blocks);
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
  builder.cost.parent_links = builder.cost.statements ? builder.cost.statements - 1 : 0;
  return ConservativeSyncAnalysis(function, builder.PrimitiveSites(), std::move(exit), builder.cost,
                                  std::move(builder.children), std::move(builder.domains),
                                  retain_incoming_diagnostics, builder.disjoint_gm_, enable_refresh_kill,
                                  std::move(selection));
}
}  // namespace pypto::ir::sync
