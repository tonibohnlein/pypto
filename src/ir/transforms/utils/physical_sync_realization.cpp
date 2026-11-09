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
#include <array>
#include <cctype>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/backend_handler.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/core/logging.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/pass_context.h"
#include "pypto/ir/transforms/pass_properties.h"
#include "pypto/ir/transforms/passes.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/sync_placement.h"

namespace pypto::ir::sync {
namespace {
using Direction = std::pair<PipeType, PipeType>;
using Family = PhysicalSyncFamily;

Occurrence Source(const Family& f, int64_t ordinal) {
  return {f.source_region, f.source_first + ordinal, f.source_site};
}
Occurrence Target(const Family& f, int64_t ordinal) {
  return {f.target_region, f.target_first + ordinal, f.target_site};
}
Direction Pipes(const CompactSyncResult& a, const Family& f) {
  return {a.segments[f.source_region].pipes[f.source_site], a.segments[f.target_region].pipes[f.target_site]};
}

// This is a narrow binding of the existing native resource declarations, not
// an inference that arbitrary missing catalog entries mean signal-free code.
// Direct Buffer IR emits TLOAD(Mat/Vec), TSTORE(Acc/Vec), TMOV(Mat->Left/Right),
// TEXTRACT(Mat->Left/Right), TMATMUL and TMATMUL_ACC for these admitted calls.
// In manual a2a3 mode those variants have no hidden AIC flags. TEXTRACT's
// catalogued V<->S claims belong to the excluded Vec overload. Still consume
// any applicable fixed claims for the owning core conservatively, and refuse
// unresolved owners. Vec arithmetic takes its macro from the shared recipe.
std::map<Direction, std::set<int>> NativeClaims(const CompactSyncResult& a, const std::string& target) {
  CHECK(target == "a2a3") << "Unsupported physical target: native variants are qualified only for a2a3";
  auto backend = backend::GetBackendInstance(backend::BackendType::Ascend910);
  std::map<Direction, std::set<int>> claimed;
  std::set<std::string> checked;
  for (const auto& segment : a.input.segments) {
    if (!segment.count) continue;
    for (const auto& site : segment.sites) {
      for (const auto& call : {site.original, site.continuation}) {
        if (!call) continue;
        std::string macro;
        if (IsOp(call, "buffer.load")) {
          macro = "pto::TLOAD";
        } else if (IsOp(call, "buffer.store")) {
          macro = "pto::TSTORE";
        } else if (IsOp(call, "buffer.copy")) {
          macro = "pto::TMOV";
        } else if (IsOp(call, "buffer.extract")) {
          macro = "pto::TEXTRACT";
        } else if (IsOp(call, "buffer.matmul")) {
          macro = "pto::TMATMUL";
        } else if (IsOp(call, "buffer.matmul_acc")) {
          macro = "pto::TMATMUL_ACC";
        } else {
          // The same recipe selects the emitted PTO operation. Resource claims
          // remain authoritative; unresolved ISA-helper ownership still refuses.
          const char* native = nullptr;
          if (auto recipe = backend::FindBufferElementwiseRecipe(call->op_->name_)) {
            native = recipe->native_op;
          } else if (auto recipe = backend::FindBufferRowRecipe(call->op_->name_)) {
            native = recipe->native_op;
          }
          CHECK(native) << "Unavailable native synchronization resource binding: " << call->op_->name_;
          macro = std::string(native).substr(4);
          std::transform(macro.begin(), macro.end(), macro.begin(),
                         [](unsigned char c) { return std::toupper(c); });
          macro = "pto::" + macro;
        }
        if (!checked.insert(macro).second) continue;
        const auto declaration = backend->FindIsaMacroResources(macro);
        if (!declaration) continue;  // Only the enumerated, audited native variants.
        for (const auto& claim : declaration->claims) {
          using pypto::sync::CoreScope;
          using pypto::sync::IsaIdSource;
          using pypto::sync::IsaResourceLifetime;
          using pypto::sync::IsaSyncMode;
          using pypto::sync::ResourceKind;
          if (std::find(claim.modes.begin(), claim.modes.end(), IsaSyncMode::Manual) == claim.modes.end())
            continue;
          if (macro == "pto::TEXTRACT" && claim.pool.core == CoreScope::AIV) continue;
          CHECK(claim.pool.kind == ResourceKind::PipelineEvent &&
                claim.pool.core ==
                    (a.input.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC) &&
                claim.source == IsaIdSource::Fixed && claim.lifetime == IsaResourceLifetime::Invocation)
              << "Unavailable native flag owner/variant binding: " << macro;
          for (auto id : claim.ids) claimed[{claim.pool.src, claim.pool.dst}].insert(static_cast<int>(id));
        }
      }
    }
  }
  return claimed;
}

class Allocator {
 public:
  Allocator(const CompactSyncResult& analysis, const std::string& target,
            const std::optional<std::vector<int>>& restricted)
      : a_(analysis),
        catalog_(pypto::sync::FindResourceCatalog(target)),
        restricted_(restricted),
        claimed_(NativeClaims(analysis, target)) {
    CHECK(catalog_) << "Unavailable synchronization resource catalog: " << target;
    CHECK(a_.input.hardware_target == target)
        << "Synchronization hardware contract differs from lowering target";
    if (restricted_) {
      std::set<int> unique;
      for (int id : *restricted_)
        CHECK(id >= 0 && unique.insert(id).second) << "Invalid/duplicate eligible ID";
    }
  }
  PhysicalSyncRealization Run() {
    for (const auto& [direction, ids] : claimed_)
      for (int id : ids) result_.native_reserved_ids.emplace_back(direction.first, direction.second, id);
    for (size_t r = 0; r < a_.segments.size(); ++r) {
      const auto& segment = a_.segments[r];
      for (const auto& f : segment.demands) {
        CHECK(f.distance == 0 || f.distance == 1) << "Unavailable physical displacement";
        if (segment.count <= f.distance) continue;
        result_.families.push_back({r, f.source, r, f.target, 0, f.distance, segment.count - f.distance, {}});
      }
    }
    for (const auto& [source, target] : a_.crossing_demands) {
      auto [sr, sv, ss] = source;
      auto [tr, tv, ts] = target;
      CHECK(sr < tr && sv == a_.segments[sr].count - 1 && tv == 0) << "Unavailable physical crossing frame";
      result_.families.push_back({sr, ss, tr, ts, sv, tv, 1, {}});
    }
    QualifyBarriers();
    for (size_t index = 0; index < result_.families.size(); ++index) {
      auto& f = result_.families[index];
      const auto direction = Pipes(a_, f);
      if (direction.first == direction.second) continue;
      const auto& ids = Eligible(direction);
      int64_t width = f.count;
      if (f.count > 1) {
        result_.finite_minimum = false;
        const auto& segment = a_.segments[f.source_region];
        INTERNAL_CHECK(f.source_region == f.target_region && !segment.distance.empty())
            << "Internal error: repeated family without a periodic owner";
        const auto distance = segment.distance[2 * f.target_site + 1][2 * f.source_site];
        // Canonical direct placement associates WAIT(h)->SET(g) with
        // C_consumer(h)->I_source(g). Native wrap padding makes this distance
        // uniform for every valid visit. Unique lanes cover shorter prefixes.
        if (distance >= 0) {
          const __int128 period = __int128(f.target_first) - f.source_first + distance;
          width = static_cast<int64_t>(std::min<__int128>(f.count, std::max<__int128>(1, period)));
        }
      }
      CHECK(width <= static_cast<int64_t>(ids.size()))
          << (f.count == 1 ? "Fixed-plan scarcity: direction has no eligible ID"
                           : "Allocation unproved: family lane bound exceeds eligible capacity")
          << " direction=" << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
          << " lanes=" << width << " eligible=" << ids.size();
      f.ids.resize(width, -1);
      for (int64_t residue = 0; residue < width; ++residue) {
        const int64_t last = residue + ((f.count - 1 - residue) / width) * width;
        lanes_[direction].push_back(
            {index, static_cast<size_t>(residue), Source(f, residue), Target(f, last)});
        ++result_.reserved_lanes;
      }
    }
    for (auto& [direction, lanes] : lanes_) Color(direction, lanes);
    return std::move(result_);
  }

 private:
  struct Lane {
    size_t family, residue;
    Occurrence first, last;
  };
  const CompactSyncResult& a_;
  const pypto::sync::ResourceCatalog* catalog_;
  const std::optional<std::vector<int>>& restricted_;
  std::map<Direction, std::set<int>> claimed_;
  std::map<Direction, std::vector<int>> eligible_;
  std::map<Direction, std::vector<Lane>> lanes_;
  PhysicalSyncRealization result_;
  bool Reaches(const Occurrence& a, const Occurrence& b) {
    ++result_.reuse_queries;
    return a_.Precedes(a, true, b, false);
  }
  const std::vector<int>& Eligible(Direction d) {
    auto [it, fresh] = eligible_.try_emplace(d);
    if (!fresh) return it->second;
    using pypto::sync::CoreScope;
    using pypto::sync::ResourceKind;
    const auto core = a_.input.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC;
    auto spec = catalog_->FindPool({ResourceKind::PipelineEvent, core, d.first, d.second});
    CHECK(spec) << "Unavailable directional event pool";
    auto legal = spec->legal_ids;
    if (legal.empty()) {
      legal.resize(spec->capacity);
      std::iota(legal.begin(), legal.end(), 0);
    }
    std::set<int> available;
    for (auto id : legal)
      if (std::find(spec->reserved_ids.begin(), spec->reserved_ids.end(), id) == spec->reserved_ids.end() &&
          !claimed_[d].count(static_cast<int>(id)))
        available.insert(static_cast<int>(id));
    if (restricted_) {
      for (int id : *restricted_) {
        CHECK(id < static_cast<int>(spec->capacity) &&
              std::find(legal.begin(), legal.end(), id) != legal.end())
            << "Requested ID is outside the legal target pool";
        if (available.count(id)) it->second.push_back(id);
      }
    } else {
      it->second.assign(available.begin(), available.end());
    }
    return it->second;
  }
  void QualifyBarriers() {
    // Predecessor lookup is indexed by region and pipe. For repeated bodies,
    // check both startup and steady-state; no trip count is unfolded.
    std::vector<std::array<std::optional<Occurrence>, kPipeCount>> before(a_.segments.size());
    std::array<std::optional<Occurrence>, kPipeCount> previous{};
    std::vector<std::vector<std::optional<size_t>>> local(a_.segments.size());
    std::vector<std::array<std::optional<size_t>, kPipeCount>> last(a_.segments.size());
    for (size_t r = 0; r < a_.segments.size(); ++r) {
      before[r] = previous;
      local[r].resize(a_.segments[r].pipes.size());
      if (!a_.segments[r].count) continue;
      for (size_t s = 0; s < a_.segments[r].pipes.size(); ++s) {
        auto pipe = a_.segments[r].pipes[s];
        local[r][s] = last[r][pipe];
        last[r][pipe] = s;
        previous[pipe] = Occurrence{r, a_.segments[r].count - 1, s};
      }
    }
    for (const auto& f : result_.families) {
      auto [src, dst] = Pipes(a_, f);
      if (src != dst) continue;
      CHECK(dst != PipeType::S) << "Scalar barriers are forbidden; require qualified native scalar order";
      CHECK(dst == PipeType::M || dst == PipeType::V) << "Unavailable same-pipe barrier mechanism";
      for (int64_t offset : std::set<int64_t>{0, f.count - 1}) {
        auto [r, v, s] = Target(f, offset);
        std::optional<Occurrence> pred;
        if (local[r][s])
          pred = Occurrence{r, v, *local[r][s]};
        else if (v)
          pred = Occurrence{r, v - 1, *last[r][dst]};
        else
          pred = before[r][dst];
        CHECK(pred && Reaches(*pred, Target(f, offset)))
            << "Unavailable order-equal same-pipe barrier placement";
      }
    }
  }
  void Color(Direction direction, std::vector<Lane>& lanes) {
    // Sort two boundary chains once, then sweep the monotone first-reuse
    // suffix. Queries are charged separately (the retained regional oracle
    // need not be constant-time). No quadratic directory scan or K unfolding.
    auto by_source = [](const Lane& x, const Lane& y) {
      return std::tie(x.first, x.family, x.residue) < std::tie(y.first, y.family, y.residue);
    };
    std::sort(lanes.begin(), lanes.end(), by_source);
    std::vector<size_t> waits(lanes.size()), release(lanes.size());
    std::iota(waits.begin(), waits.end(), 0);
    std::sort(waits.begin(), waits.end(), [&](size_t x, size_t y) {
      return std::tie(lanes[x].last, lanes[x].family, lanes[x].residue) <
             std::tie(lanes[y].last, lanes[y].family, lanes[y].residue);
    });
    size_t next = 0;
    for (auto i : waits) {
      while (next < lanes.size() && !Reaches(lanes[i].last, lanes[next].first)) ++next;
      release[i] = next;
      INTERNAL_CHECK(next > i) << "Internal error: cyclic physical lane span";
    }
    std::vector<std::vector<int>> buckets(lanes.size() + 1);
    std::vector<int> free;
    const auto& eligible = Eligible(direction);
    size_t colors = 0;
    for (size_t i = 0; i < lanes.size(); ++i) {
      free.insert(free.end(), buckets[i].begin(), buckets[i].end());
      if (free.empty()) {
        CHECK(colors < eligible.size())
            << (result_.finite_minimum ? "Fixed-plan scarcity"
                                       : "Allocation unproved by whole-span reservation")
            << ": direction=" << PipeTypeToString(direction.first) << "->"
            << PipeTypeToString(direction.second) << " overlap_at_publication=" << i
            << " required_at_least=" << colors + 1 << " eligible=" << eligible.size();
        free.push_back(eligible[colors++]);
      }
      int id = free.back();
      free.pop_back();
      const auto& lane = lanes[i];
      result_.families[lane.family].ids[lane.residue] = id;
      buckets[release[i]].push_back(id);
    }
  }
};

class Emitter : public SyncPlacement {
 public:
  Emitter(const CompactSyncResult& a, PhysicalSyncRealization& result) : SyncPlacement(a), result_(result) {
    for (size_t i = 0; i < result.families.size(); ++i) {
      const auto& f = result.families[i];
      incoming_[{f.target_region, f.target_site}].push_back(i);
      if (!f.ids.empty()) outgoing_[{f.source_region, f.source_site}].push_back(i);
    }
  }

 private:
  PhysicalSyncRealization& result_;
  std::map<SyncSiteKey, std::vector<size_t>> incoming_, outgoing_;
  ExprPtr Integer(int64_t n, const Span& span) {
    return std::make_shared<ConstInt>(n, DataType::INDEX, span);
  }
  ExprPtr Ordinal(size_t region, const Span& span) {
    const auto& s = analysis_.input.segments[region];
    if (!s.original_loop) return Integer(0, span);
    ExprPtr ordinal = s.original_loop->loop_var_;
    if (s.start) ordinal = std::make_shared<Sub>(ordinal, Integer(s.start, span), DataType::INDEX, span);
    if (s.step != 1)
      ordinal = std::make_shared<FloorDiv>(ordinal, Integer(s.step, span), DataType::INDEX, span);
    return ordinal;
  }
  StmtPtr Guard(ExprPtr condition, StmtPtr body) {
    ++result_.guard_sites;
    return std::make_shared<IfStmt>(condition, body, std::nullopt, std::vector<VarPtr>{}, body->span_);
  }
  StmtPtr Command(const Family& f, bool publish, const Span& span) {
    const auto [src, dst] = Pipes(analysis_, f);
    const size_t r = publish ? f.source_region : f.target_region;
    const int64_t first = publish ? f.source_first : f.target_first;
    const auto ordinal = Ordinal(r, span);
    ExprPtr index = ordinal;
    if (first) index = std::make_shared<Sub>(index, Integer(first, span), DataType::INDEX, span);
    std::vector<StmtPtr> commands;
    if (src == dst) {
      INTERNAL_CHECK(dst == PipeType::M || dst == PipeType::V)
          << "Only qualified M/V barriers may be emitted; scalar barriers are forbidden";
      auto call = OpRegistry::GetInstance().Create(dst == PipeType::M ? "system.bar_m" : "system.bar_v", {},
                                                   {}, span);
      commands.push_back(std::make_shared<EvalStmt>(call, span));
      ++result_.barrier_sites;
    } else {
      ExprPtr residue;
      if (f.ids.size() > 1)
        residue = std::make_shared<FloorMod>(index, Integer(f.ids.size(), span), DataType::INDEX, span);
      for (size_t i = 0; i < f.ids.size(); ++i) {
        auto call = OpRegistry::GetInstance().Create(
            publish ? "system.set_flag" : "system.wait_flag", {},
            {{"src_pipe", static_cast<int>(src)},
             {"dst_pipe", static_cast<int>(dst)},
             {"event_id", f.ids[i]},
             {"core_type",
              std::string(analysis_.input.owner->func_type_ == FunctionType::AIV ? "aiv" : "aic")}},
            span);
        StmtPtr command = std::make_shared<EvalStmt>(call, span);
        if (residue)
          command = Guard(std::make_shared<Eq>(residue, Integer(i, span), DataType::BOOL, span), command);
        commands.push_back(command);
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      }
    }
    StmtPtr body = std::make_shared<SeqStmts>(commands, span);
    // The range guard surrounds residue arithmetic: no absent predecessor is
    // evaluated/consumed. Final-only exports never publish on earlier visits.
    if (first + f.count < analysis_.segments[r].count)
      body = Guard(std::make_shared<Lt>(ordinal, Integer(first + f.count, span), DataType::BOOL, span), body);
    if (first) body = Guard(std::make_shared<Ge>(ordinal, Integer(first, span), DataType::BOOL, span), body);
    return body;
  }
  StmtPtr Wrap(const StmtPtr& payload, SyncSiteKey site) override {
    std::vector<StmtPtr> statements;
    // Barrier precedes incoming WAITs. Outgoing SETs remain immediately after
    // their payload, hence before the next payload's barrier and WAIT group.
    for (auto i : incoming_[site])
      if (result_.families[i].ids.empty())
        statements.push_back(Command(result_.families[i], false, payload->span_));
    for (auto i : incoming_[site])
      if (!result_.families[i].ids.empty())
        statements.push_back(Command(result_.families[i], false, payload->span_));
    statements.push_back(payload);
    for (auto i : outgoing_[site]) statements.push_back(Command(result_.families[i], true, payload->span_));
    return std::make_shared<SeqStmts>(statements, payload->span_);
  }
};
}  // namespace
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids) {
  auto result = Allocator(analysis, target, eligible_ids).Run();
  Emitter emitter(analysis, result);
  result.function = emitter.VisitFunction(analysis.input.owner);
  return result;
}
}  // namespace pypto::ir::sync

namespace pypto::ir::pass {
namespace {
FunctionPtr SyncOutcome(const FunctionPtr& function, bool pypto, const std::string& reason) {
  auto attrs = function->attrs_;
  attrs.erase(std::remove_if(attrs.begin(), attrs.end(),
                             [](const auto& attr) {
                               return attr.first == kAttrPyptoAutosync ||
                                      attr.first == kAttrAutosyncFallbackReason;
                             }),
              attrs.end());
  attrs.emplace_back(kAttrPyptoAutosync, pypto);
  if (!pypto) attrs.emplace_back(kAttrAutosyncFallbackReason, reason);
  return std::make_shared<Function>(function->name_, function->params_, function->param_directions_,
                                    function->return_types_, function->body_, function->span_,
                                    function->func_type_, function->level_, function->role_, attrs,
                                    function->requires_runtime_binding_, function->ir_stage_);
}
}  // namespace
Pass RealizeSync() {
  auto transform = [](const FunctionPtr& function) -> FunctionPtr {
    if (!IsInCoreType(function->func_type_)) return function;
    // These functions do not enter PTO code generation; their external owner
    // remains responsible for synchronization. Do not claim PTOAS delegation.
    if (function->requires_runtime_binding_ || function->HasAttr(kAttrExternalSource) ||
        function->HasAttr(kAttrBuiltinTemplateDir))
      return function;
    auto* ctx = PassContext::Current();
    CHECK(ctx && ctx->GetEnablePyptoAutoSync()) << "RealizeSync requires enable_pypto_autosync=True";
    CHECK(!function->GetAttr<bool>(kAttrPyptoAutosync))
        << "RealizeSync cannot reprocess an owned physical plan";
    if (function->func_type_ != FunctionType::AIC && function->func_type_ != FunctionType::AIV)
      return SyncOutcome(function, false, "PyPTO synchronization supports only AIC/AIV leaves");
    if (function->ir_stage_ != FunctionIRStage::Buffer || ctx->GetMemoryPlanner() == MemoryPlanner::PtoAS)
      return SyncOutcome(function, false, "PyPTO synchronization requires physically placed Buffer IR");
    if (!ctx->GetAutoSyncDisjointGM())
      return SyncOutcome(function, false, "Disjoint GM argument contract was not supplied");
    auto target = ctx->GetBackendHandler()->GetPtoTargetArch();
    if (target != "a2a3")
      return SyncOutcome(function, false, "PyPTO physical synchronization supports only manual a2a3 macros");
    // Only publish after complete analysis AND realization. A qualification or
    // resource refusal returns the original body, never partially emitted IR.
    // Internal errors are deliberately not caught. The strict retained-result
    // APIs still throw on these user-facing input/proof refusals.
    try {
      auto analysis = sync::AnalyzeCompactSync(sync::ImportPlacedSync(function, true, target));
      auto physical = sync::RealizePhysicalSync(analysis, target, std::nullopt);
      // PTOAS recognizes literal flags, not empty plans or barriers, as manual
      // ownership. Do not fabricate a dummy flag or mix partial ownership.
      if (!physical.set_sites || !physical.wait_sites)
        return SyncOutcome(function, false, "Candidate has no SET/WAIT marker; delegate the original kernel");
      return SyncOutcome(physical.function, true, "");
    } catch (const pypto::ValueError& error) {
      return SyncOutcome(function, false, error.what());
    }
  };
  return CreateFunctionPass(transform, "RealizeSync", PassProperties{});
}
}  // namespace pypto::ir::pass
