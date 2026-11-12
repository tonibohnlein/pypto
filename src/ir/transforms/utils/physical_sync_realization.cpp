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
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/backend/common/backend.h"
#include "pypto/backend/common/buffer_elementwise_recipes.h"
#include "pypto/backend/common/buffer_indexed_recipes.h"
#include "pypto/backend/common/buffer_row_recipes.h"
#include "pypto/core/logging.h"
#include "pypto/ir/arith/analyzer.h"
#include "pypto/ir/op_registry.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "pypto/ir/transforms/utils/compact_sync_analysis.h"
#include "pypto/sync/isa_macro_resources.h"
#include "pypto/sync/resource_catalog.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_placement.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {
using Direction = std::pair<PipeType, PipeType>;
using Family = PhysicalSyncFamily;

Occurrence Endpoint(const Family& f, int64_t ordinal, bool publish) {
  auto visit = publish ? f.source_first : f.target_first;
  if (f.axes.empty()) {
    visit += ordinal;
  } else {
    for (auto axis = f.axes.rbegin(); axis != f.axes.rend(); ++axis) {
      visit += (ordinal % axis->count) * (publish ? axis->source_stride : axis->target_stride);
      ordinal /= axis->count;
    }
    INTERNAL_CHECK(!ordinal) << "Internal error: scoped physical record exceeds domain";
  }
  return {publish ? f.source_region : f.target_region, visit, publish ? f.source_site : f.target_site};
}
Occurrence Source(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, true); }
Occurrence Target(const Family& f, int64_t ordinal) { return Endpoint(f, ordinal, false); }
Direction Pipes(const CompactSyncInput& a, const Family& f) {
  return {a.segments[f.source_region].sites[f.source_site].pipe,
          a.segments[f.target_region].sites[f.target_site].pipe};
}

// This is a narrow binding of the existing native resource declarations, not
// an inference that arbitrary missing catalog entries mean signal-free code.
// Direct Buffer IR emits TLOAD(Mat/Vec), TSTORE(Acc/Vec), TMOV(Mat->Left/Right),
// TEXTRACT(Mat->Left/Right), TMATMUL and TMATMUL_ACC for these admitted calls.
// In manual a2a3 mode those variants have no hidden AIC flags. TEXTRACT's
// catalogued V<->S claims belong to the excluded Vec overload. Still consume
// any applicable fixed claims for the owning core conservatively, and refuse
// unresolved owners. Vec arithmetic takes its macro from the shared recipe.
std::map<Direction, std::set<int>> NativeClaims(const CompactSyncInput& a, const std::string& target) {
  CHECK(target == "a2a3") << "Unsupported physical target: native variants are qualified only for a2a3";
  auto backend = backend::GetBackendInstance(backend::BackendType::Ascend910);
  std::map<Direction, std::set<int>> claimed;
  std::set<std::string> checked;
  for (const auto& segment : a.segments) {
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
          } else if (auto recipe = backend::FindBufferIndexedRecipe(call->op_->name_)) {
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
                    (a.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC) &&
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
  Allocator(const CompactSyncInput& input, std::vector<Family> families, const DirectSyncOrder& order,
            const std::string& target, const std::optional<std::vector<int>>& restricted)
      : input_(input),
        order_(order),
        catalog_(pypto::sync::FindResourceCatalog(target)),
        restricted_(restricted),
        claimed_(SyncNativeClaims(input, target)) {
    result_.families = std::move(families);
    result_.finite_minimum = order_.finite_minimum && !input_.region_tree;
    CHECK(order_.precedes && order_.reference_before && order_.threshold)
        << "Missing direct synchronization order proof";
    for (const auto& f : result_.families) {
      CHECK(f.source_region < input_.segments.size() && f.target_region < input_.segments.size())
          << "Invalid direct family region";
      const auto& source = input_.segments[f.source_region];
      const auto& target_segment = input_.segments[f.target_region];
      CHECK(f.source_site < source.sites.size() && f.target_site < target_segment.sites.size() &&
            f.source_first >= 0 && f.target_first >= 0 && f.count > 0 && f.ids.empty())
          << "Invalid direct family endpoint or preassigned IDs";
      __int128 last_source = f.source_first, last_target = f.target_first, volume = 1;
      if (f.axes.empty()) {
        last_source += f.count - 1;
        last_target += f.count - 1;
      } else {
        CHECK(f.axes.size() <= source.nested_frames.size() &&
              f.axes.size() <= target_segment.nested_frames.size())
            << "Invalid direct family arity";
        for (size_t i = 0; i < f.axes.size(); ++i) {
          const auto& axis = f.axes[i];
          CHECK(axis.frame == source.nested_frames[i].loop &&
                axis.frame == target_segment.nested_frames[i].loop && axis.count > 0 &&
                axis.count <= source.nested_frames[i].count &&
                axis.count <= target_segment.nested_frames[i].count && axis.source_stride >= 0 &&
                axis.target_stride >= 0)
              << "Invalid direct family scope";
          volume *= axis.count;
          CHECK(volume <= std::numeric_limits<int64_t>::max()) << "Direct family volume overflow";
          last_source += __int128(axis.count - 1) * axis.source_stride;
          last_target += __int128(axis.count - 1) * axis.target_stride;
          CHECK(last_source < source.count && last_target < target_segment.count)
              << "Direct family axis exceeds occurrence domain";
        }
        CHECK(volume == f.count) << "Inconsistent direct family volume";
      }
      CHECK(last_source < source.count && last_target < target_segment.count)
          << "Direct family endpoint exceeds occurrence domain";
    }
    CHECK(catalog_) << "Unavailable synchronization resource catalog: " << target;
    CHECK(input_.hardware_target == target)
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
    QualifyBarriers();
    for (size_t index = 0; index < result_.families.size(); ++index) {
      auto& f = result_.families[index];
      const auto direction = Pipes(input_, f);
      if (direction.first == direction.second) continue;
      const auto& ids = Eligible(direction);
      int64_t width = f.count;
      if (f.count > 1 && input_.region_tree) {
        result_.finite_minimum = false;
        CHECK(order_.scoped_reuse && order_.scoped_reuse(index))
            << "Scoped period-one physical reuse is unproved; no capacity obstruction established";
        width = 1;
      } else if (f.count > 1) {
        result_.finite_minimum = false;
        INTERNAL_CHECK(f.source_region == f.target_region)
            << "Internal error: repeated family without a periodic owner";
        const auto distance = order_.threshold(f.source_region, f.target_site, f.source_site);
        // Canonical direct placement associates WAIT(h)->SET(g) with
        // C_consumer(h)->I_source(g). Reference-forward witnesses between valid
        // endpoints stay inside the finite prefix. Clip before adding a wide
        // threshold, so huge reachable distances never overflow or mean absence.
        const int64_t displacement = f.target_first - f.source_first;
        INTERNAL_CHECK(displacement >= 0) << "Negative direct-family displacement";
        if (distance && *distance < f.count && displacement < f.count) {
          const __int128 period = __int128(displacement) + *distance;
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
  const CompactSyncInput& input_;
  const DirectSyncOrder& order_;
  const pypto::sync::ResourceCatalog* catalog_;
  const std::optional<std::vector<int>>& restricted_;
  std::map<Direction, std::set<int>> claimed_;
  std::map<Direction, std::vector<int>> eligible_;
  std::map<Direction, std::vector<Lane>> lanes_;
  PhysicalSyncRealization result_;
  bool Reaches(const Occurrence& a, const Occurrence& b) {
    ++result_.reuse_queries;
    return order_.precedes(a, b);
  }
  const std::vector<int>& Eligible(Direction d) {
    auto [it, fresh] = eligible_.try_emplace(d);
    if (!fresh) return it->second;
    it->second = SyncEligibleIds(input_, input_.hardware_target, d, restricted_, claimed_);
    return it->second;
  }
  void QualifyBarriers() {
    if (input_.region_tree) {
      for (size_t i = 0; i < result_.families.size(); ++i) {
        const auto [source, target] = Pipes(input_, result_.families[i]);
        if (source != target) continue;
        CHECK(target == PipeType::M || target == PipeType::V)
            << "Scoped same-pipe barrier mechanism is unavailable; scalar barriers are forbidden";
        CHECK(order_.scoped_barrier && order_.scoped_barrier(i))
            << "Scoped same-pipe barrier adjacency is unproved";
      }
      return;
    }
    // Predecessor lookup is indexed by region and pipe. For repeated bodies,
    // check both startup and steady-state; no trip count is unfolded.
    std::vector<std::array<std::optional<Occurrence>, kPipeCount>> before(input_.segments.size());
    std::array<std::optional<Occurrence>, kPipeCount> previous{};
    std::vector<std::vector<std::optional<size_t>>> local(input_.segments.size());
    std::vector<std::array<std::optional<size_t>, kPipeCount>> last(input_.segments.size());
    for (size_t r = 0; r < input_.segments.size(); ++r) {
      before[r] = previous;
      local[r].resize(input_.segments[r].sites.size());
      if (!input_.segments[r].count) continue;
      for (size_t s = 0; s < input_.segments[r].sites.size(); ++s) {
        auto pipe = input_.segments[r].sites[s].pipe;
        local[r][s] = last[r][pipe];
        last[r][pipe] = s;
        previous[pipe] = Occurrence{r, input_.segments[r].count - 1, s};
      }
    }
    for (const auto& f : result_.families) {
      auto [src, dst] = Pipes(input_, f);
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
    auto by_source = [&](const Lane& x, const Lane& y) {
      if (x.first != y.first) return order_.reference_before(x.first, y.first);
      return std::tie(x.family, x.residue) < std::tie(y.family, y.residue);
    };
    std::sort(lanes.begin(), lanes.end(), by_source);
    const auto colors = ColorOrderedSyncLanes(
        lanes.size(), direction,
        [&](size_t x, size_t y) {
          if (lanes[x].last != lanes[y].last) return order_.reference_before(lanes[x].last, lanes[y].last);
          return std::tie(lanes[x].family, lanes[x].residue) < std::tie(lanes[y].family, lanes[y].residue);
        },
        [&](size_t x, size_t y) { return Reaches(lanes[x].last, lanes[y].first); }, Eligible(direction),
        result_.finite_minimum);
    for (size_t i = 0; i < lanes.size(); ++i) {
      const auto& lane = lanes[i];
      result_.families[lane.family].ids[lane.residue] = colors[i];
    }
  }
};

// Equality is optional optimization evidence. An unsupported comparison must
// preserve the accepted unfactored plan, rather than become an export refusal.
bool SameControl(const IRNodePtr& left, const IRNodePtr& right) {
  try {
    return structural_equal(left, right, false);
  } catch (const StructuralComparisonUnsupported&) {
    return false;
  }
}

// Only generated predicates enter this simplifier. Original payload/control
// expressions are never rewritten. Bounds come from the retained occurrence
// frame, with a fresh analyzer per cut so facts cannot leak across resets.
class CommandConditions : public IRMutator {
  arith::Analyzer analyzer_;
  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    ExprPtr condition = branch->condition_;
    try {
      condition = analyzer_.Simplify(condition);
    } catch (const StructuralComparisonUnsupported&) {
      // Arithmetic simplification is optional; retained realization stays valid.
    }
    if (auto known = As<ConstBool>(condition)) {
      if (known->value_) return VisitStmt(branch->then_body_);
      return branch->else_body_ ? VisitStmt(*branch->else_body_)
                                : std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, branch->span_);
    }
    return std::make_shared<IfStmt>(
        condition, VisitStmt(branch->then_body_),
        branch->else_body_ ? std::optional<StmtPtr>(VisitStmt(*branch->else_body_)) : std::nullopt,
        branch->return_vars_, branch->span_);
  }

 public:
  explicit CommandConditions(const SyncSegment& segment) {
    auto bind = [&](const ForStmtPtr& loop) {
      const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
                 step = As<ConstInt>(loop->step_);
      if (start && stop && step && step->value_ > 0 && start->value_ < stop->value_)
        analyzer_.Bind(loop->loop_var_, start->value_, stop->value_);
    };
    if (!segment.nested_frames.empty()) {
      for (const auto& frame : segment.nested_frames) bind(frame.loop);
    } else if (segment.original_loop) {
      bind(segment.original_loop);
    }
  }
};

// Scope-local value numbering for generated command predicates. Count only
// strict, pure expression trees in conditions at this sequence level. Insert
// a definition at its FIRST evaluation, never at a loop preheader or outside
// a guard. Children have independent tables; no branch-private value escapes.
// Hash collisions lose sharing rather than inducing an all-pairs search.
class CommandValues : public IRMutator {
  SyncTokenNames& names_;
  PhysicalSyncRealization& work_;
  struct Entry {
    ExprPtr expression;
    size_t uses = 0;
    VarPtr value;
  };
  using Table = std::map<uint64_t, Entry>;
  static bool Candidate(const ExprPtr& expression) {
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return false;
    if (As<FloorDiv>(expression) && As<Var>(binary->left_) && As<ConstInt>(binary->right_)) return true;
    const auto type = As<ScalarType>(expression->GetType());
    const auto left = As<ScalarType>(binary->left_->GetType());
    const auto right = As<ScalarType>(binary->right_->GetType());
    // Hash whole comparisons and constant-size normalized IV quotients only.
    // Hashing every nested arithmetic subtree would hide quadratic traversal.
    return type && left && right && type->dtype_ == DataType::BOOL && left->dtype_ == DataType::INDEX &&
           right->dtype_ == DataType::INDEX;
  }
  bool Pure(const ExprPtr& expression) {
    ++work_.control_expression_visits;
    const auto type = As<ScalarType>(expression->GetType());
    if (!type || (type->dtype_ != DataType::INDEX && type->dtype_ != DataType::BOOL)) return false;
    if (As<Var>(expression) || As<ConstInt>(expression) || As<ConstBool>(expression)) return true;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary || As<And>(expression) || As<Or>(expression)) return false;
    if (As<FloorDiv>(expression) || As<FloorMod>(expression)) {
      auto divisor = As<ConstInt>(binary->right_);
      if (!divisor || divisor->value_ <= 0) return false;
    }
    return Pure(binary->left_) && Pure(binary->right_);
  }
  void Count(const ExprPtr& expression, Table& table) {
    ++work_.control_expression_visits;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return;
    if (Candidate(expression)) {
      const auto key = structural_hash(expression, false);
      auto& entry = table[key];
      if (!entry.expression) entry.expression = expression;
      if (SameControl(entry.expression, expression)) ++entry.uses;
    }
    Count(binary->left_, table);
    Count(binary->right_, table);
  }
  class Values : public IRMutator {
    Table& table_;
    SyncTokenNames& names_;
    std::vector<StmtPtr>& definitions_;
    PhysicalSyncRealization& work_;

   public:
    Values(Table& table, SyncTokenNames& names, std::vector<StmtPtr>& definitions,
           PhysicalSyncRealization& work)
        : table_(table), names_(names), definitions_(definitions), work_(work) {}
    ExprPtr VisitBinaryExpr_(const BinaryExprPtr& expression) override {
      ++work_.control_expression_visits;
      auto found = Candidate(expression) ? table_.find(structural_hash(expression, false)) : table_.end();
      Entry* entry =
          found != table_.end() && found->second.uses > 1 && SameControl(found->second.expression, expression)
              ? &found->second
              : nullptr;
      if (entry && entry->value) return entry->value;
      auto rewritten = IRMutator::VisitBinaryExpr_(expression);
      if (!entry) return rewritten;
      entry->value =
          std::make_shared<Var>(names_.Next("sync_control_"), expression->GetType(), expression->span_);
      definitions_.push_back(std::make_shared<AssignStmt>(entry->value, rewritten, expression->span_));
      ++work_.shared_control_values;
      return entry->value;
    }
  };
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    Table table;
    for (const auto& statement : sequence->stmts_)
      if (auto branch = As<IfStmt>(statement); branch && Pure(branch->condition_))
        Count(branch->condition_, table);
    std::vector<StmtPtr> output;
    for (const auto& statement : sequence->stmts_) {
      if (auto branch = As<IfStmt>(statement); branch && Pure(branch->condition_)) {
        const auto condition = Values(table, names_, output, work_).VisitExpr(branch->condition_);
        output.push_back(std::make_shared<IfStmt>(
            condition, VisitStmt(branch->then_body_),
            branch->else_body_ ? std::optional<StmtPtr>(VisitStmt(*branch->else_body_)) : std::nullopt,
            branch->return_vars_, branch->span_));
      } else {
        output.push_back(VisitStmt(statement));
      }
    }
    return std::make_shared<SeqStmts>(output, sequence->span_);
  }

 public:
  CommandValues(SyncTokenNames& names, PhysicalSyncRealization& work) : names_(names), work_(work) {}
};

std::vector<StmtPtr> MaterializeCommandValues(const std::vector<StmtPtr>& commands, SyncTokenNames& names,
                                              PhysicalSyncRealization& work) {
  const auto sequence = std::make_shared<SeqStmts>(commands, Span::unknown());
  const auto previous = work.shared_control_values;
  try {
    return As<SeqStmts>(CommandValues(names, work).VisitStmt(sequence))->stmts_;
  } catch (const StructuralComparisonUnsupported&) {
    work.shared_control_values = previous;
    return commands;  // Optional value numbering cannot invalidate a plan.
  }
}

size_t CountCommandGuards(const StmtPtr& statement) {
  size_t result = 0;
  if (auto sequence = As<SeqStmts>(statement))
    for (const auto& child : sequence->stmts_) result += CountCommandGuards(child);
  if (auto branch = As<IfStmt>(statement)) {
    result += 1 + CountCommandGuards(branch->then_body_);
    if (branch->else_body_) result += CountCommandGuards(*branch->else_body_);
  }
  return result;
}

// Only emitter-created forests enter this builder: no payloads, definitions,
// effectful conditions or result-bearing branches. Share consecutive equal
// guard prefixes, preserving every leaf and its order. Mutable child vectors
// avoid repeatedly copying a growing immutable body. Work is linear in visited
// control occurrences plus structural-comparison traversal (not distinct DAGs).
class CommandGuards {
  struct Node {
    ExprPtr condition;
    StmtPtr leaf;
    std::vector<Node> children;
  };
  std::vector<Node> roots_;
  size_t shared_ = 0, visits_ = 0;

  void Append(std::vector<Node>& nodes, const StmtPtr& statement) {
    ++visits_;
    if (auto sequence = As<SeqStmts>(statement)) {
      for (const auto& child : sequence->stmts_) Append(nodes, child);
    } else if (auto branch = As<IfStmt>(statement);
               branch && !branch->else_body_ && branch->return_vars_.empty()) {
      if (nodes.empty() || !nodes.back().condition ||
          !SameControl(nodes.back().condition, branch->condition_)) {
        nodes.push_back({branch->condition_, nullptr, {}});
      } else {
        ++shared_;
      }
      Append(nodes.back().children, branch->then_body_);
    } else {
      // In particular, ID decision chains are opaque. Sharing must not move
      // a later command into one arm, or collect A/B/A into A/A/B.
      nodes.push_back({nullptr, statement, {}});
    }
  }
  static std::vector<StmtPtr> Freeze(const std::vector<Node>& nodes) {
    std::vector<StmtPtr> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes) {
      if (node.condition) {
        const auto span = node.condition->span_;
        auto body = std::make_shared<SeqStmts>(Freeze(node.children), span);
        result.push_back(
            std::make_shared<IfStmt>(node.condition, body, std::nullopt, std::vector<VarPtr>{}, span));
      } else {
        result.push_back(node.leaf);
      }
    }
    return result;
  }

 public:
  std::vector<StmtPtr> Build(const std::vector<StmtPtr>& commands, PhysicalSyncRealization& result) {
    for (const auto& command : commands) Append(roots_, command);
    INTERNAL_CHECK(result.guard_sites >= shared_) << "Internal error: command guard accounting";
    result.guard_sites -= shared_;
    result.shared_guard_sites += shared_;
    result.control_statement_visits += visits_;
    return Freeze(roots_);
  }
};

using CaseCommands = SyncCommandCase;
struct PhysicalCuts {
  std::vector<StmtPtr> before, after;
  std::vector<CaseCommands> before_cases, after_cases;
  // Upper bound on the invocation coordinates reaching this ORIGINAL cut.
  // Only a retained zero-visit proof removes coordinates. Missing evidence
  // leaves the full domain; a present cut with no commands is never absence.
  std::vector<std::pair<int64_t, int64_t>> domain;
};
// Compare each newly supplied complete sequence only with its predecessor.
// The accumulated interval is constant-size; its body is never concatenated.
// Missing cases remain numeric gaps, even when both neighboring bodies match.
void AppendCase(std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> blocks,
                const std::vector<StmtPtr>& commands, const Span& span) {
  if (commands.empty()) return;
  auto body = std::make_shared<SeqStmts>(commands, span);
  if (!cases.empty() && cases.back().blocks.second < std::numeric_limits<int64_t>::max() &&
      cases.back().blocks.second + 1 == blocks.first && SameControl(cases.back().body, body)) {
    cases.back().blocks.second = blocks.second;
  } else {
    cases.push_back({blocks, body});
  }
}
using CutInventory = std::map<const Call*, std::shared_ptr<PhysicalCuts>>;

using Intervals = std::vector<std::pair<int64_t, int64_t>>;
using AbsentLoops = std::map<const ForStmt*, Intervals>;

void RetainAbsentLoops(const SyncRegion& region, std::pair<int64_t, int64_t> blocks, AbsentLoops& absent) {
  if (region.repetition && !region.repetition->count) {
    auto& ranges = absent[region.repetition->loop.get()];
    // Cases arrive in increasing order; retain a normalized union without
    // visiting any dynamic occurrence or the body of an absent loop.
    if (!ranges.empty() && ranges.back().second + 1 == blocks.first)
      ranges.back().second = blocks.second;
    else
      ranges.push_back(blocks);
    return;
  }
  for (const auto& child : region.children) RetainAbsentLoops(child, blocks, absent);
}

Intervals SubtractIntervals(const Intervals& domain, const Intervals& absent) {
  Intervals result;
  size_t j = 0;
  for (auto [lo, hi] : domain) {
    while (j < absent.size() && absent[j].second < lo) ++j;
    for (size_t k = j; k < absent.size() && absent[k].first <= hi; ++k) {
      if (lo < absent[k].first) result.emplace_back(lo, absent[k].first - 1);
      lo = std::max(lo, absent[k].second + 1);
      if (lo > hi) break;
    }
    if (lo <= hi) result.emplace_back(lo, hi);
  }
  return result;
}

// One original-tree traversal transports supplied absence evidence to cuts.
// Work includes the interval pieces propagated through nested scopes, not just
// distinct source nodes. It is independent of launch size and loop trip counts.
class CutDomains : public IRVisitor {
  const AbsentLoops& absent_;
  CutInventory& cuts_;
  Intervals domain_;
  PhysicalSyncRealization& work_;
  void VisitStmt_(const ForStmtPtr& loop) override {
    auto saved = domain_;
    work_.cut_domain_interval_visits += domain_.size();
    if (auto found = absent_.find(loop.get()); found != absent_.end()) {
      work_.cut_domain_interval_visits += found->second.size();
      domain_ = SubtractIntervals(domain_, found->second);
    }
    VisitStmt(loop->body_);
    domain_ = std::move(saved);
  }
  void VisitStmt_(const AssignStmtPtr&) override {}
  void VisitStmt_(const IfStmtPtr& branch) override {
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (auto found = cuts_.find(call.get()); found != cuts_.end()) {
      work_.cut_domain_interval_visits += domain_.size();
      found->second->domain = domain_;
    }
  }

 public:
  CutDomains(const AbsentLoops& absent, CutInventory& cuts, std::pair<int64_t, int64_t> blocks,
             PhysicalSyncRealization& work)
      : absent_(absent), cuts_(cuts), domain_{blocks}, work_(work) {
    for (auto& [anchor, cut] : cuts_) cut->domain = domain_;
  }
};

bool DomainIntersects(const Intervals& domain, int64_t lo, int64_t hi) {
  if (lo > hi) return false;
  const auto next =
      std::lower_bound(domain.begin(), domain.end(), lo,
                       [](const auto& interval, int64_t value) { return interval.second < value; });
  return next != domain.end() && next->first <= hi;
}

std::vector<CaseCommands> CoalesceCutCases(const std::vector<CaseCommands>& source, const Intervals& domain,
                                           PhysicalSyncRealization& work) {
  std::vector<CaseCommands> result;
  for (const auto& item : source) {
    ++work.cut_case_comparisons;
    if (!result.empty() &&
        !DomainIntersects(domain, result.back().blocks.second + 1, item.blocks.first - 1) &&
        SameControl(result.back().body, item.body)) {
      result.back().blocks.second = item.blocks.second;
      ++work.coalesced_cut_cases;
    } else {
      result.push_back(item);
    }
  }
  return result;
}

class Emitter : public SyncPlacement {
 public:
  Emitter(const CompactSyncInput& input, PhysicalSyncRealization& result)
      : SyncPlacement(input), result_(result) {
    for (size_t i = 0; i < result.families.size(); ++i) {
      const auto& f = result.families[i];
      incoming_[{f.target_region, f.target_site}].push_back(i);
      if (!f.ids.empty()) outgoing_[{f.source_region, f.source_site}].push_back(i);
    }
  }

  void AppendCuts(CutInventory& inventory, std::pair<int64_t, int64_t> blocks) {
    for (size_t r = 0; r < input_.segments.size(); ++r) {
      const auto& region = input_.segments[r];
      if (!region.count) continue;
      for (size_t s = 0; s < region.sites.size(); ++s) {
        const auto& site = region.sites[s];
        auto commands = Commands({r, s}, site.original->span_);
        if (commands.before.empty() && commands.after.empty()) continue;
        auto& retained = inventory[site.original.get()];
        if (!retained) retained = std::make_shared<PhysicalCuts>();
        if (site.continuation) {
          auto& continuation = inventory[site.continuation.get()];
          INTERNAL_CHECK(!continuation || continuation == retained)
              << "Internal error: incompatible original initialization cut";
          continuation = retained;
        }
        AppendCase(retained->before_cases, blocks, commands.before, site.original->span_);
        AppendCase(retained->after_cases, blocks, commands.after, site.original->span_);
      }
    }
  }

 private:
  PhysicalSyncRealization& result_;
  std::unique_ptr<SyncTokenNames> names_;
  std::map<SyncSiteKey, std::vector<size_t>> incoming_, outgoing_;
  ExprPtr Integer(int64_t n, const Span& span) {
    return std::make_shared<ConstInt>(n, DataType::INDEX, span);
  }
  ExprPtr Ordinal(size_t region, const Span& span) {
    const auto& s = input_.segments[region];
    if (!s.original_loop) return Integer(0, span);
    if (!s.nested_frames.empty()) {
      ExprPtr ordinal = Integer(0, span);
      for (const auto& frame : s.nested_frames) {
        ExprPtr digit = frame.loop->loop_var_;
        if (frame.start)
          digit = std::make_shared<Sub>(digit, Integer(frame.start, span), DataType::INDEX, span);
        if (frame.step != 1)
          digit = std::make_shared<FloorDiv>(digit, Integer(frame.step, span), DataType::INDEX, span);
        ordinal = std::make_shared<Add>(
            std::make_shared<Mul>(ordinal, Integer(frame.count, span), DataType::INDEX, span), digit,
            DataType::INDEX, span);
      }
      return ordinal;
    }
    ExprPtr ordinal = s.original_loop->loop_var_;
    if (s.start) ordinal = std::make_shared<Sub>(ordinal, Integer(s.start, span), DataType::INDEX, span);
    if (s.step != 1)
      ordinal = std::make_shared<FloorDiv>(ordinal, Integer(s.step, span), DataType::INDEX, span);
    return ordinal;
  }
  ExprPtr FrameDigit(const SyncLoopFrame& frame, const Span& span) {
    // This original loop has exactly one admitted visit regardless of its
    // invocation-dependent start. No normalized placeholder is read as an IV.
    if (frame.count == 1) return Integer(0, span);
    ExprPtr value = frame.loop->loop_var_;
    if (frame.start_expression || frame.start)
      value = std::make_shared<Sub>(
          value, frame.start_expression ? frame.start_expression : Integer(frame.start, span),
          DataType::INDEX, span);
    if (frame.step != 1)
      value = std::make_shared<FloorDiv>(value, Integer(frame.step, span), DataType::INDEX, span);
    return value;
  }
  std::vector<ExprPtr> ScopedGuards(const Family& family, bool publish, const Span& span) {
    const auto region = publish ? family.source_region : family.target_region;
    auto base = publish ? family.source_first : family.target_first;
    const auto& frames = input_.segments[region].nested_frames;
    std::vector<int64_t> digits(frames.size());
    for (size_t i = frames.size(); i-- > 0;) {
      INTERNAL_CHECK(frames[i].count > 0) << "Internal error: physical endpoint in absent frame";
      digits[i] = base % frames[i].count;
      base /= frames[i].count;
    }
    INTERNAL_CHECK(!base && family.axes.size() <= frames.size()) << "Internal error: physical scope arity";
    std::vector<ExprPtr> guards;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto count = i < family.axes.size() ? family.axes[i].count : 1;
      if (i < family.axes.size())
        INTERNAL_CHECK(family.axes[i].frame == frames[i].loop) << "Internal error: physical scope identity";
      if (!digits[i] && count == frames[i].count) continue;
      const auto value = FrameDigit(frames[i], span);
      if (count == 1) {
        guards.push_back(std::make_shared<Eq>(value, Integer(digits[i], span), DataType::BOOL, span));
      } else {
        if (digits[i])
          guards.push_back(std::make_shared<Ge>(value, Integer(digits[i], span), DataType::BOOL, span));
        if (digits[i] + count < frames[i].count)
          guards.push_back(
              std::make_shared<Lt>(value, Integer(digits[i] + count, span), DataType::BOOL, span));
      }
    }
    return guards;
  }
  StmtPtr Guard(ExprPtr condition, StmtPtr body) {
    ++result_.guard_sites;
    return std::make_shared<IfStmt>(condition, body, std::nullopt, std::vector<VarPtr>{}, body->span_);
  }
  StmtPtr Command(const Family& f, bool publish, const Span& span) {
    const auto [src, dst] = Pipes(input_, f);
    const size_t r = publish ? f.source_region : f.target_region;
    const int64_t first = publish ? f.source_first : f.target_first;
    const bool scoped = input_.region_tree.has_value();
    // The initial scoped certificate deliberately proves period one. Keep
    // physical table coordinates separate from the full logical record tuple.
    INTERNAL_CHECK(!scoped || f.ids.size() <= 1) << "Internal error: unsupported scoped physical period";
    const auto ordinal = scoped ? Integer(0, span) : Ordinal(r, span);
    ExprPtr index = ordinal;
    if (!scoped && first) index = std::make_shared<Sub>(index, Integer(first, span), DataType::INDEX, span);
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
      StmtPtr selection;
      // Distinct residue tests are mutually exclusive. Keep the last equality
      // too: no extra exhaustiveness/range premise is needed for this rewrite.
      // All selection arithmetic stays beneath the original presence guards.
      for (size_t i = f.ids.size(); i-- > 0;) {
        auto call = OpRegistry::GetInstance().Create(
            publish ? "system.set_flag" : "system.wait_flag", {},
            {{"src_pipe", static_cast<int>(src)},
             {"dst_pipe", static_cast<int>(dst)},
             {"event_id", f.ids[i]},
             {"core_type", std::string(input_.owner->func_type_ == FunctionType::AIV ? "aiv" : "aic")}},
            span);
        StmtPtr command = std::make_shared<EvalStmt>(call, span);
        if (residue) {
          ++result_.guard_sites;
          command = std::make_shared<IfStmt>(
              std::make_shared<Eq>(residue, Integer(i, span), DataType::BOOL, span), command,
              selection ? std::optional<StmtPtr>(selection) : std::nullopt, std::vector<VarPtr>{}, span);
        }
        selection = command;
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      }
      if (selection) commands.push_back(selection);
    }
    StmtPtr body = std::make_shared<SeqStmts>(commands, span);
    // The range guard surrounds residue arithmetic: no absent predecessor is
    // evaluated/consumed. Final-only exports never publish on earlier visits.
    if (scoped) {
      auto guards = ScopedGuards(f, publish, span);
      for (auto guard = guards.rbegin(); guard != guards.rend(); ++guard) body = Guard(*guard, body);
      return body;
    }
    if (first + f.count < input_.segments[r].count)
      body = Guard(std::make_shared<Lt>(ordinal, Integer(first + f.count, span), DataType::BOOL, span), body);
    if (first) body = Guard(std::make_shared<Ge>(ordinal, Integer(first, span), DataType::BOOL, span), body);
    return body;
  }
  PhysicalCuts Commands(SyncSiteKey site, const Span& span) {
    PhysicalCuts cuts;
    const auto previous_guards = result_.guard_sites;
    // Preserve the common direct protocol: local barrier, incoming WAITs,
    // payload, outgoing SETs. Case aggregation never interleaves active cases.
    for (auto i : incoming_[site])
      if (result_.families[i].ids.empty()) cuts.before.push_back(Command(result_.families[i], false, span));
    for (auto i : incoming_[site])
      if (!result_.families[i].ids.empty()) cuts.before.push_back(Command(result_.families[i], false, span));
    for (auto i : outgoing_[site]) cuts.after.push_back(Command(result_.families[i], true, span));
    CommandConditions conditions(input_.segments[site.first]);
    size_t remaining = 0;
    for (auto* commands : {&cuts.before, &cuts.after})
      for (auto& statement : *commands) {
        statement = conditions.VisitStmt(statement);
        remaining += CountCommandGuards(statement);
      }
    INTERNAL_CHECK(result_.guard_sites >= previous_guards + remaining) << "Unexpected new command guards";
    result_.simplified_guard_sites += result_.guard_sites - previous_guards - remaining;
    result_.guard_sites = previous_guards + remaining;
    cuts.before = ShareSyncCommandGuards(cuts.before, result_);
    cuts.after = ShareSyncCommandGuards(cuts.after, result_);
    return cuts;
  }
  StmtPtr Wrap(const StmtPtr& payload, SyncSiteKey site) override {
    auto cuts = Commands(site, payload->span_);
    if (!names_) names_ = std::make_unique<SyncTokenNames>(input_.owner);
    cuts.before = MaterializeCommandValues(cuts.before, *names_, result_);
    cuts.after = MaterializeCommandValues(cuts.after, *names_, result_);
    cuts.before.push_back(payload);
    cuts.before.insert(cuts.before.end(), cuts.after.begin(), cuts.after.end());
    return std::make_shared<SeqStmts>(cuts.before, payload->span_);
  }
};
// The command builder above owns all protocol mechanics. This adapter only
// installs already-qualified case cuts in the unchanged original payload tree.
class InvocationEmitter : public IRMutator {
  const CutInventory& inventory_;
  SyncTokenNames names_;
  PhysicalSyncRealization& work_;
  StmtPtr Wrap(const StmtPtr& payload, const PhysicalCuts& cuts) {
    auto statements = MaterializeCommandValues(cuts.before, names_, work_);
    statements.push_back(payload);
    const auto after = MaterializeCommandValues(cuts.after, names_, work_);
    statements.insert(statements.end(), after.begin(), after.end());
    return std::make_shared<SeqStmts>(statements, payload->span_);
  }
  static const Call* ArmCall(StmtPtr arm) {
    if (auto seq = As<SeqStmts>(arm)) {
      if (seq->stmts_.size() != 1) return nullptr;
      arm = seq->stmts_.front();
    }
    const auto eval = As<EvalStmt>(arm);
    return eval ? As<Call>(eval->expr_).get() : nullptr;
  }
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    const auto found = inventory_.find(call.get());
    return found == inventory_.end() ? statement : Wrap(statement, *found->second);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& statement) override {
    const auto found = inventory_.find(ArmCall(statement->then_body_));
    if (found == inventory_.end()) return IRMutator::VisitStmt_(statement);
    CHECK(statement->else_body_) << "Invocation cut requires the original initialization alternatives";
    const auto other = inventory_.find(ArmCall(*statement->else_body_));
    CHECK(other != inventory_.end() && other->second == found->second)
        << "Invocation initialization alternatives have incompatible command cuts";
    return Wrap(statement, *found->second);
  }

 public:
  InvocationEmitter(const CutInventory& inventory, const FunctionPtr& owner, PhysicalSyncRealization& work)
      : inventory_(inventory), names_(owner), work_(work) {}
};

std::vector<Family> DirectFamilies(const CompactSyncResult& analysis) {
  PhysicalSyncRealization result_;
  if (analysis.input.region_tree) {
    result_.finite_minimum = false;  // Sufficient whole-span assignment, not a minimum-ID proof.
    const auto owned = analysis.ExportScopedDemands();
    CHECK(owned) << "Scoped endpoint export is unavailable";
    for (const auto& schema : owned->families) {
      const auto [sr, sv, ss] = schema.source;
      const auto [tr, tv, ts] = schema.target;
      __int128 count = 1;
      for (const auto& axis : schema.axes) {
        count *= axis.count;
        CHECK(count <= std::numeric_limits<int64_t>::max()) << "Scoped physical record domain overflow";
      }
      result_.families.push_back({sr, ss, tr, ts, sv, tv, static_cast<int64_t>(count), {}, schema.axes});
    }
  } else {
    const auto crossings = analysis.ExportCrossingDemands();
    CHECK(crossings.has_value())
        << "Minimum demands are retained; shared-repetition physical endpoint scope is unavailable";
    for (size_t r = 0; r < analysis.segments.size(); ++r) {
      const auto& segment = analysis.segments[r];
      for (const auto& f : segment.demands) {
        CHECK(f.distance >= 0 && f.source < segment.pipes.size() && f.target < segment.pipes.size())
            << "Invalid physical demand family";
        if (segment.count <= f.distance) continue;
        result_.families.push_back({r, f.source, r, f.target, 0, f.distance, segment.count - f.distance, {}});
      }
    }
    for (const auto& [source, target] : *crossings) {
      auto [sr, sv, ss] = source;
      auto [tr, tv, ts] = target;
      CHECK(sr < analysis.segments.size() && tr < analysis.segments.size() && sr < tr)
          << "Unavailable physical crossing frame";
      CHECK(sv >= 0 && sv < analysis.segments[sr].count && tv >= 0 && tv < analysis.segments[tr].count &&
            ss < analysis.segments[sr].pipes.size() && ts < analysis.segments[tr].pipes.size())
          << "Invalid physical crossing occurrence";
      result_.families.push_back({sr, ss, tr, ts, sv, tv, 1, {}});
    }
  }
  return std::move(result_.families);
}
DirectSyncOrder MathematicalOrder(const CompactSyncResult& analysis) {
  return {[&](const Occurrence& a, const Occurrence& b) { return analysis.Precedes(a, true, b, false); },
          [&](const Occurrence& a, const Occurrence& b) { return analysis.ReferenceBefore(a, b); },
          [&](size_t r, size_t a, size_t b) { return analysis.segments[r].Threshold(a, true, b, false); },
          [&](size_t i) { return analysis.ScopedConsecutiveReuse(i); },
          [&](size_t i) { return analysis.ScopedAdjacentBarrier(i); },
          true};
}

}  // namespace
StmtPtr MaterializeSyncCommandValues(const FunctionPtr& owner, const StmtPtr& commands) {
  SyncTokenNames names(owner);
  PhysicalSyncRealization work;
  return std::make_shared<SeqStmts>(MaterializeCommandValues({commands}, names, work), commands->span_);
}
std::vector<StmtPtr> EmitSyncCommandCases(const ExprPtr& block, const Intervals& possible,
                                          const std::vector<SyncCommandCase>& cases,
                                          PhysicalSyncRealization& work, const Span& span) {
  std::vector<StmtPtr> target;
  if (possible.empty()) return target;
  int64_t previous = -1;
  for (const auto& [lo, hi] : possible) {
    CHECK(lo > previous && hi >= lo && hi < std::numeric_limits<int64_t>::max())
        << "Invalid cut execution domain";
    previous = hi;
  }
  previous = -1;
  for (const auto& item : cases) {
    CHECK(item.blocks.first > previous && item.blocks.second >= item.blocks.first &&
          item.blocks.second < std::numeric_limits<int64_t>::max())
        << "Invalid ordered command cases";
    previous = item.blocks.second;
  }
  const auto context = std::make_pair(possible.front().first, possible.back().second);
  for (const auto& item : CoalesceCutCases(cases, possible, work)) {
    const auto selected = std::make_pair(std::max(context.first, item.blocks.first),
                                         std::min(context.second, item.blocks.second));
    if (selected.first > selected.second) continue;
    const auto guard = SyncInvocationGuard(block, context, selected, span);
    target.push_back(
        guard ? std::make_shared<IfStmt>(guard, item.body, std::nullopt, std::vector<VarPtr>{}, span)
              : item.body);
  }
  return target;
}
std::vector<StmtPtr> ShareSyncCommandGuards(const std::vector<StmtPtr>& commands,
                                            PhysicalSyncRealization& counters) {
  return CommandGuards().Build(commands, counters);
}
SyncNativeReservations SyncNativeClaims(const CompactSyncInput& input, const std::string& target) {
  return NativeClaims(input, target);
}
std::vector<int> SyncEligibleIds(const CompactSyncInput& input, const std::string& target,
                                 SyncDirection direction, const std::optional<std::vector<int>>& restricted,
                                 const SyncNativeReservations& claimed) {
  using pypto::sync::CoreScope;
  using pypto::sync::ResourceKind;
  const auto* catalog = pypto::sync::FindResourceCatalog(target);
  CHECK(catalog) << "Unavailable synchronization resource catalog: " << target;
  CHECK(input.hardware_target == target) << "Synchronization hardware contract differs from lowering target";
  const auto core = input.owner->func_type_ == FunctionType::AIV ? CoreScope::AIV : CoreScope::AIC;
  auto spec = catalog->FindPool({ResourceKind::PipelineEvent, core, direction.first, direction.second});
  CHECK(spec) << "Unavailable directional event pool";
  auto legal = spec->legal_ids;
  if (legal.empty()) {
    legal.resize(spec->capacity);
    std::iota(legal.begin(), legal.end(), 0);
  }
  std::set<int> available;
  const auto owners = claimed.find(direction);
  for (auto id : legal)
    if (std::find(spec->reserved_ids.begin(), spec->reserved_ids.end(), id) == spec->reserved_ids.end() &&
        (owners == claimed.end() || !owners->second.count(static_cast<int>(id))))
      available.insert(static_cast<int>(id));
  std::vector<int> result;
  if (restricted) {
    std::set<int> unique;
    for (int id : *restricted) {
      CHECK(id >= 0 && unique.insert(id).second) << "Invalid/duplicate eligible ID";
      CHECK(id < static_cast<int>(spec->capacity) && std::find(legal.begin(), legal.end(), id) != legal.end())
          << "Requested ID is outside the legal target pool";
      if (available.count(id)) result.push_back(id);
    }
  } else {
    result.assign(available.begin(), available.end());
  }
  return result;
}
std::vector<int> ColorOrderedSyncLanes(size_t count, SyncDirection direction,
                                       const std::function<bool(size_t, size_t)>& wait_before,
                                       const std::function<bool(size_t, size_t)>& reusable,
                                       const std::vector<int>& eligible, bool finite_minimum) {
  std::vector<size_t> waits(count), release(count);
  std::iota(waits.begin(), waits.end(), 0);
  std::sort(waits.begin(), waits.end(), wait_before);
  size_t next = 0;
  for (auto i : waits) {
    while (next < count && !reusable(i, next)) ++next;
    release[i] = next;
    INTERNAL_CHECK(next > i) << "Internal error: cyclic physical lane span";
  }
  std::vector<std::vector<int>> buckets(count + 1);
  std::vector<int> free, result(count);
  size_t colors = 0;
  for (size_t i = 0; i < count; ++i) {
    free.insert(free.end(), buckets[i].begin(), buckets[i].end());
    if (free.empty()) {
      CHECK(colors < eligible.size())
          << (finite_minimum ? "Fixed-plan scarcity" : "Allocation unproved by whole-span reservation")
          << ": direction=" << PipeTypeToString(direction.first) << "->" << PipeTypeToString(direction.second)
          << " overlap_at_publication=" << i << " required_at_least=" << colors + 1
          << " eligible=" << eligible.size();
      free.push_back(eligible[colors++]);
    }
    const int id = free.back();
    free.pop_back();
    result[i] = id;
    buckets[release[i]].push_back(id);
  }
  return result;
}
PhysicalSyncRealization AllocateDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                           const DirectSyncOrder& order, const std::string& target,
                                           const std::optional<std::vector<int>>& restricted) {
  return Allocator(input, std::move(families), order, target, restricted).Run();
}
PhysicalSyncRealization RealizeDirectSync(const CompactSyncInput& input, std::vector<Family> families,
                                          const DirectSyncOrder& order, const std::string& target,
                                          const std::optional<std::vector<int>>& restricted) {
  auto result = AllocateDirectSync(input, std::move(families), order, target, restricted);
  result.function = Emitter(input, result).VisitFunction(input.owner);
  return result;
}
PhysicalSyncRealization RealizePhysicalSync(const CompactSyncResult& analysis, const std::string& target,
                                            const std::optional<std::vector<int>>& eligible_ids) {
  return RealizeDirectSync(analysis.input, DirectFamilies(analysis), MathematicalOrder(analysis), target,
                           eligible_ids);
}
FunctionPtr EmitAssignedSyncCases(const FunctionPtr& owner, const SyncParameterBounds& parameters,
                                  std::pair<int64_t, int64_t> blocks,
                                  const std::vector<AssignedSyncCase>& cases) {
  CHECK(owner && !cases.empty()) << "Missing complete invocation case owner";
  const auto block = cases.size() > 1 ? DominatingSyncBlock(owner) : nullptr;
  CutInventory cuts;
  AbsentLoops absent;
  auto next = blocks.first;
  for (const auto& item : cases) {
    CHECK(item.input && item.plan && item.input->owner == owner &&
          item.input->parameter_bounds == parameters && item.input->block_bounds &&
          item.input->block_bounds->first == next && item.input->block_bounds->second >= next &&
          item.input->block_bounds->second < std::numeric_limits<int64_t>::max())
        << "Invocation cases must retain exhaustive original owners and contracts";
    next = item.input->block_bounds->second + 1;
    if (item.input->region_tree)
      RetainAbsentLoops(*item.input->region_tree, *item.input->block_bounds, absent);
    Emitter(*item.input, *item.plan).AppendCuts(cuts, *item.input->block_bounds);
  }
  CHECK(next == blocks.second + 1) << "Incomplete invocation activation domain";
  auto& control_work = *cases.front().plan;
  CutDomains(absent, cuts, blocks, control_work).VisitStmt(owner->body_);
  // Original initialization/accumulation calls can name the same cut. Freeze
  // it once, after all domains have been checked. A union guard is omitted only
  // when the identical sequence covers the cut's entire possible execution
  // domain. Unknown domains retain the previous whole-invocation behavior.
  std::set<const PhysicalCuts*> finalized;
  for (const auto& [anchor, cut] : cuts) {
    if (!finalized.insert(cut.get()).second) continue;
    auto emit = [&](const std::vector<CaseCommands>& source, std::vector<StmtPtr>& target) {
      target = EmitSyncCommandCases(block, cut->domain, source, control_work, owner->span_);
    };
    emit(cut->before_cases, cut->before);
    emit(cut->after_cases, cut->after);
  }
  // Exclusive cases share palettes only within this original invocation.
  // Guarded cuts contain endpoint/ID arithmetic as well as SET/WAIT commands.
  return InvocationEmitter(cuts, owner, control_work).VisitFunction(owner);
}
PhysicalSyncInvocation RealizePhysicalInvocation(const SyncInvocationAnalysis& analysis,
                                                 const std::string& target,
                                                 const std::optional<std::vector<int>>& eligible_ids) {
  // Qualify executable case selection before spending work on allocation.
  CHECK(analysis.cases.size() <= 1 || DominatingSyncBlock(analysis.owner))
      << "Original dominating invocation block binding is unavailable";
  PhysicalSyncInvocation result{analysis, nullptr, {}};
  result.cases.reserve(analysis.cases.size());
  std::vector<AssignedSyncCase> cases;
  for (const auto& item : analysis.cases) {
    CHECK(item.mathematics &&
          item.mathematics->input.block_bounds == std::make_pair(item.block_first, item.block_last))
        << "Missing original invocation analysis";
    result.cases.push_back(AllocateDirectSync(item.mathematics->input, DirectFamilies(*item.mathematics),
                                              MathematicalOrder(*item.mathematics), target, eligible_ids));
    cases.push_back({&item.mathematics->input, &result.cases.back()});
  }
  result.function =
      EmitAssignedSyncCases(analysis.owner, analysis.parameter_bounds, analysis.block_bounds, cases);
  return result;
}

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
}  // namespace pypto::ir::sync
