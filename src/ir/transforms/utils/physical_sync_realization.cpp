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
// frame. Each region caches an analyzer containing only immutable original-loop
// bounds; it never learns branch-local assumptions or a previous visit's values.
class CommandConditions {
  arith::Analyzer analyzer_;

 public:
  ExprPtr Simplify(const ExprPtr& expression) {
    try {
      return analyzer_.Simplify(expression);
    } catch (const StructuralComparisonUnsupported&) {
      return expression;
    }
  }
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

// Lowering-owned control descriptors. No scalar statement is created while
// competing schedules are built or costed. Each forest belongs to one original
// payload cut and side; expressions retain original SSA/frame identity. Leaves
// retain the allocator's exact action, direction and literal ID in their Call.
struct CommandControl;
using ControlPtr = std::shared_ptr<const CommandControl>;
using ControlForest = std::vector<ControlPtr>;
struct CommandControl {
  ExprPtr condition, command;
  ControlForest body, alternative;
  Span span;
};
ControlPtr ControlSequence(ControlForest body, const Span& span, PhysicalSyncRealization& work) {
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(CommandControl{nullptr, nullptr, std::move(body), {}, span});
}
ControlPtr ControlBranch(ExprPtr condition, ControlPtr body, std::optional<ControlPtr> alternative,
                         const Span& span, PhysicalSyncRealization& work) {
  if (auto known = As<ConstBool>(condition))
    return known->value_ ? body : alternative ? *alternative : ControlSequence({}, span, work);
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(
      CommandControl{std::move(condition),
                     nullptr,
                     {std::move(body)},
                     alternative ? ControlForest{*alternative} : ControlForest{},
                     span});
}
ControlPtr ControlCommand(ExprPtr command, const Span& span, PhysicalSyncRealization& work) {
  ++work.control_descriptors;
  return std::make_shared<CommandControl>(CommandControl{nullptr, std::move(command), {}, {}, span});
}
bool SameControl(const ControlPtr& a, const ControlPtr& b) {
  if (a == b) return true;
  if (static_cast<bool>(a->condition) != static_cast<bool>(b->condition) ||
      static_cast<bool>(a->command) != static_cast<bool>(b->command) || a->body.size() != b->body.size() ||
      a->alternative.size() != b->alternative.size())
    return false;
  if (a->condition && !SameControl(a->condition, b->condition)) return false;
  if (a->command && !SameControl(a->command, b->command)) return false;
  for (size_t i = 0; i < a->body.size(); ++i)
    if (!SameControl(a->body[i], b->body[i])) return false;
  for (size_t i = 0; i < a->alternative.size(); ++i)
    if (!SameControl(a->alternative[i], b->alternative[i])) return false;
  return true;
}
ControlPtr ImportControl(const StmtPtr& stmt, PhysicalSyncRealization& work) {
  if (auto sequence = As<SeqStmts>(stmt)) {
    ControlForest children;
    for (const auto& child : sequence->stmts_) children.push_back(ImportControl(child, work));
    return ControlSequence(std::move(children), stmt->span_, work);
  }
  if (auto branch = As<IfStmt>(stmt)) {
    CHECK(branch->return_vars_.empty()) << "Command schedule cannot move definitions";
    return ControlBranch(branch->condition_, ImportControl(branch->then_body_, work),
                         branch->else_body_
                             ? std::optional<ControlPtr>(ImportControl(*branch->else_body_, work))
                             : std::nullopt,
                         stmt->span_, work);
  }
  auto command = As<EvalStmt>(stmt);
  CHECK(command) << "Command schedule requires command-only input";
  return ControlCommand(command->expr_, stmt->span_, work);
}
std::vector<StmtPtr> MaterializeControl(const ControlForest& forest, PhysicalSyncRealization& work) {
  std::vector<StmtPtr> result;
  for (const auto& node : forest) {
    if (node->command) {
      ++work.materialized_control_statements;
      result.push_back(std::make_shared<EvalStmt>(node->command, node->span));
    } else {
      auto body = MaterializeControl(node->body, work);
      auto sequence = [&](std::vector<StmtPtr> statements) -> StmtPtr {
        if (statements.size() == 1 && As<SeqStmts>(statements.front())) return statements.front();
        ++work.materialized_control_statements;
        return std::make_shared<SeqStmts>(std::move(statements), node->span);
      };
      if (!node->condition) {
        result.push_back(sequence(std::move(body)));
      } else {
        std::optional<StmtPtr> alternative;
        if (!node->alternative.empty()) alternative = sequence(MaterializeControl(node->alternative, work));
        ++work.materialized_control_statements;
        result.push_back(std::make_shared<IfStmt>(node->condition, sequence(std::move(body)), alternative,
                                                  std::vector<VarPtr>{}, node->span));
      }
    }
  }
  return result;
}
void VisitControlExpressions(const ControlPtr& node, IRVisitor& visitor) {
  if (node->condition) visitor.VisitExpr(node->condition);
  if (node->command) visitor.VisitExpr(node->command);
  for (const auto& child : node->body) VisitControlExpressions(child, visitor);
  for (const auto& child : node->alternative) VisitControlExpressions(child, visitor);
}

// Share adjacent guard prefixes directly in descriptors. ID decision chains
// remain opaque; no command crosses an alternative or a payload boundary.
class CommandGuards {
  struct Node {
    ExprPtr condition;
    ControlPtr leaf;
    std::vector<Node> children;
  };
  std::vector<Node> roots_;
  size_t shared_ = 0, visits_ = 0;
  void Append(std::vector<Node>& nodes, const ControlPtr& statement) {
    ++visits_;
    if (!statement->condition && !statement->command) {
      for (const auto& child : statement->body) Append(nodes, child);
    } else if (statement->condition && statement->alternative.empty()) {
      if (nodes.empty() || !nodes.back().condition ||
          !SameControl(nodes.back().condition, statement->condition))
        nodes.push_back({statement->condition, nullptr, {}});
      else
        ++shared_;
      for (const auto& child : statement->body) Append(nodes.back().children, child);
    } else {
      nodes.push_back({nullptr, statement, {}});
    }
  }
  static ControlForest Freeze(const std::vector<Node>& nodes, PhysicalSyncRealization& work) {
    ControlForest result;
    for (const auto& node : nodes) {
      if (node.condition)
        result.push_back(ControlBranch(
            node.condition, ControlSequence(Freeze(node.children, work), node.condition->span_, work),
            std::nullopt, node.condition->span_, work));
      else
        result.push_back(node.leaf);
    }
    return result;
  }

 public:
  ControlForest Build(const ControlForest& commands, PhysicalSyncRealization& result) {
    for (const auto& command : commands) Append(roots_, command);
    INTERNAL_CHECK(result.guard_sites >= shared_) << "Internal error: command guard accounting";
    result.guard_sites -= shared_;
    result.shared_guard_sites += shared_;
    result.control_statement_visits += visits_;
    return Freeze(roots_, result);
  }
};

struct CaseCommands {
  std::pair<int64_t, int64_t> blocks;
  ControlPtr body;
};

struct PhysicalCuts {
  ControlForest before, after;
  std::vector<CaseCommands> before_cases, after_cases;
  // Upper bound on the invocation coordinates reaching this ORIGINAL cut.
  // Only a retained zero-visit proof removes coordinates. Missing evidence
  // leaves the full domain; a present cut with no commands is never absence.
  std::vector<std::pair<int64_t, int64_t>> domain;
  std::set<const Var*> uniform_variables;
  bool uniform_checked = false;
};
// Compare each newly supplied complete sequence only with its predecessor.
// The accumulated interval is constant-size; its body is never concatenated.
// Missing cases remain numeric gaps, even when both neighboring bodies match.
void AppendCase(std::vector<CaseCommands>& cases, std::pair<int64_t, int64_t> blocks,
                const ControlForest& commands, const Span& span, PhysicalSyncRealization& work) {
  if (commands.empty()) return;
  auto body = ControlSequence(commands, span, work);
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
  std::set<const Var*> uniform_variables_;
  size_t conditional_depth_ = 0;
  void VisitStmt_(const ForStmtPtr& loop) override {
    auto saved = domain_;
    work_.cut_domain_interval_visits += domain_.size();
    if (auto found = absent_.find(loop.get()); found != absent_.end()) {
      work_.cut_domain_interval_visits += found->second.size();
      domain_ = SubtractIntervals(domain_, found->second);
    }
    const auto start = As<ConstInt>(loop->start_), stop = As<ConstInt>(loop->stop_),
               step = As<ConstInt>(loop->step_);
    const bool uniform = start && stop && step && step->value_ > 0;
    auto saved_variables = uniform_variables_;
    if (!uniform) uniform_variables_.clear();
    if (uniform) uniform_variables_.insert(loop->loop_var_.get());
    VisitStmt(loop->body_);
    uniform_variables_ = std::move(saved_variables);
    domain_ = std::move(saved);
  }
  void VisitStmt_(const AssignStmtPtr&) override {}
  void VisitStmt_(const IfStmtPtr& branch) override {
    ++conditional_depth_;
    VisitStmt(branch->then_body_);
    if (branch->else_body_) VisitStmt(*branch->else_body_);
    --conditional_depth_;
  }
  void VisitStmt_(const EvalStmtPtr& statement) override {
    const auto call = As<Call>(statement->expr_);
    if (auto found = cuts_.find(call.get()); found != cuts_.end()) {
      work_.cut_domain_interval_visits += domain_.size();
      found->second->domain = domain_;
      // A branch could correlate a loop digit with the launch coordinate.
      // Without that branch, literal-bound IV domains are identical in every
      // launch case that reaches this cut. Do not infer this from sample runs.
      auto& cut = *found->second;
      const auto qualified = conditional_depth_ ? std::set<const Var*>{} : uniform_variables_;
      if (!cut.uniform_checked) {
        cut.uniform_variables = qualified;
      } else {
        for (auto it = cut.uniform_variables.begin(); it != cut.uniform_variables.end();) {
          if (!qualified.count(*it))
            it = cut.uniform_variables.erase(it);
          else
            ++it;
        }
      }
      cut.uniform_checked = true;
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

void FlattenControlSequence(const ControlPtr& node, ControlForest& commands, PhysicalSyncRealization& work) {
  ++work.case_sequence_visits;
  if (!node->condition && !node->command) {
    for (const auto& child : node->body) FlattenControlSequence(child, commands, work);
  } else {
    commands.push_back(node);
  }
}

// The tree only rearranges mutually exclusive case decisions at ONE original
// cut. Prefix/suffix commands shared by every descendant execute once, in their
// original order. Missing executing cases are explicit empty leaves; only a
// proved-absent interval may be omitted. Inner guarded commands remain opaque,
// so skipped predicates, SSA identity, command multiplicity and IDs survive.
//
// Balanced splitting bounds depth by log(C). Each level inspects at most the
// supplied sequence entries: O(S log(C+1)) plus structural comparison work on
// their opaque trees. No LCS/all-pairs search, dynamic visits or payload cloning.
class CaseSequenceTree {
  struct Slice {
    int64_t last;
    ControlForest commands;
    size_t begin = 0, end = 0;
  };
  std::vector<Slice> slices_;
  ExprPtr block_;
  Span span_;
  PhysicalSyncRealization& work_;

  void Append(int64_t first, int64_t last, const ControlPtr& body, const Intervals& possible) {
    if (!DomainIntersects(possible, first, last)) return;
    Slice slice{last, {}};
    if (body) FlattenControlSequence(body, slice.commands, work_);
    slice.end = slice.commands.size();
    slices_.push_back(std::move(slice));
  }

  bool Common(size_t first, size_t last, bool suffix) {
    const auto& base = slices_[first];
    if (base.begin == base.end) return false;
    const auto& command = base.commands[suffix ? base.end - 1 : base.begin];
    for (size_t i = first + 1; i < last; ++i) {
      ++work_.case_sequence_visits;
      const auto& slice = slices_[i];
      if (slice.begin == slice.end) return false;
      ++work_.case_sequence_comparisons;
      if (!SameControl(command, slice.commands[suffix ? slice.end - 1 : slice.begin])) return false;
    }
    return true;
  }

  ControlPtr Body(ControlForest commands) const { return ControlSequence(std::move(commands), span_, work_); }

  ControlForest Build(size_t first, size_t last) {
    if (first == last) return {};
    if (last - first == 1) {
      const auto& slice = slices_[first];
      return {slice.commands.begin() + slice.begin, slice.commands.begin() + slice.end};
    }
    ControlForest result, suffix;
    while (Common(first, last, false)) {
      result.push_back(slices_[first].commands[slices_[first].begin]);
      for (size_t i = first; i < last; ++i) ++slices_[i].begin;
    }
    while (Common(first, last, true)) {
      suffix.push_back(slices_[first].commands[slices_[first].end - 1]);
      for (size_t i = first; i < last; ++i) --slices_[i].end;
    }
    const auto middle = first + (last - first) / 2;
    auto left = Build(first, middle);
    auto right = Build(middle, last);
    const auto bound = std::make_shared<ConstInt>(slices_[middle - 1].last, DataType::INDEX, span_);
    if (!left.empty()) {
      auto condition = std::make_shared<Le>(block_, bound, DataType::BOOL, span_);
      std::optional<ControlPtr> alternative;
      if (!right.empty()) alternative = Body(std::move(right));
      result.push_back(ControlBranch(condition, Body(std::move(left)), alternative, span_, work_));
    } else if (!right.empty()) {
      auto condition = std::make_shared<Gt>(block_, bound, DataType::BOOL, span_);
      result.push_back(ControlBranch(condition, Body(std::move(right)), std::nullopt, span_, work_));
    }
    result.insert(result.end(), suffix.rbegin(), suffix.rend());
    return result;
  }

 public:
  CaseSequenceTree(const ExprPtr& block, const Span& span, PhysicalSyncRealization& work)
      : block_(block), span_(span), work_(work) {}

  ControlForest Build(const std::vector<CaseCommands>& cases, const Intervals& possible) {
    const auto end = possible.back().second;
    auto next = possible.front().first;
    for (const auto& item : cases) {
      ++work_.case_sequence_visits;
      const auto first = std::max(next, item.blocks.first);
      const auto last = std::min(end, item.blocks.second);
      if (first > last) continue;
      Append(next, first - 1, nullptr, possible);
      Append(first, last, item.body, possible);
      next = last + 1;  // Validated upper bounds exclude INT64_MAX.
    }
    Append(next, end, nullptr, possible);
    return Build(0, slices_.size());
  }
};

// Accept only a Pareto improvement in static commands, conditions and expression
// occurrences. This is a code-construction metric, not a device latency model.
// Walk both candidates once; include rejected construction in the work receipt.
class CaseCodeSize {
  class Expressions : public IRVisitor {
   public:
    size_t count = 0;
    void VisitExpr(const ExprPtr& expression) override {
      ++count;
      IRVisitor::VisitExpr(expression);
    }
  };
  void Visit(const ControlPtr& node) {
    ++statements;
    if (node->command) ++commands;
    if (node->condition) ++guards;
    Expressions counter;
    if (node->condition) counter.VisitExpr(node->condition);
    if (node->command) counter.VisitExpr(node->command);
    expressions += counter.count;
    for (const auto& child : node->body) Visit(child);
    for (const auto& child : node->alternative) Visit(child);
  }

 public:
  size_t commands = 0, guards = 0, expressions = 0, statements = 0;
  explicit CaseCodeSize(const ControlForest& body) {
    for (const auto& node : body) Visit(node);
  }
  bool Improves(const CaseCodeSize& old) const {
    return commands <= old.commands && guards <= old.guards && expressions <= old.expressions &&
           (commands < old.commands || guards < old.guards || expressions < old.expressions);
  }
};

// Sufficient uniformity certificate for cancelling unchanged inner-control
// cost. Unknown SSA values (including the launch coordinate) lose this optional
// optimization. Physical calls have literal attributes and no value operands.
class UniformCommandBody : public IRVisitor {
  const std::set<const Var*>& variables_;

 public:
  bool valid = true;
  explicit UniformCommandBody(const std::set<const Var*>& variables) : variables_(variables) {}
  void VisitExpr_(const VarPtr& variable) override { valid &= variables_.count(variable.get()) != 0; }
  void VisitStmt_(const AssignStmtPtr&) override { valid = false; }
};

// Exact cost of generated launch decisions: comparisons, eager Boolean ops,
// and branch decisions each cost one. Opaque inner guards are unchanged and
// cancelled only under UniformCommandBody's original-frame certificate.
class CaseDispatchCost {
  const ExprPtr& block_;
  int64_t value_;
  size_t cost_ = 0;
  bool valid_ = true;
  std::optional<bool> Condition(const ExprPtr& expression) {
    if (auto known = As<ConstBool>(expression)) return known->value_;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return std::nullopt;
    if (As<And>(expression) || As<Or>(expression)) {
      auto left = Condition(binary->left_), right = Condition(binary->right_);
      if (!left || !right) return std::nullopt;
      ++cost_;  // Both operands are evaluated by scalar code generation.
      return As<And>(expression) ? *left && *right : *left || *right;
    }
    const auto bound = As<ConstInt>(binary->right_);
    if (binary->left_ != block_ || !bound) return std::nullopt;
    ++cost_;
    if (As<Le>(expression)) return value_ <= bound->value_;
    if (As<Lt>(expression)) return value_ < bound->value_;
    if (As<Ge>(expression)) return value_ >= bound->value_;
    if (As<Gt>(expression)) return value_ > bound->value_;
    if (As<Eq>(expression)) return value_ == bound->value_;
    valid_ = false;
    return std::nullopt;
  }
  void Visit(const ControlPtr& stmt) {
    if (!stmt->condition) {
      for (const auto& child : stmt->body) Visit(child);
    } else {
      const auto saved = cost_;
      auto selected = Condition(stmt->condition);
      if (!selected) {
        cost_ = saved;
        return;
      }
      ++cost_;
      for (const auto& child : *selected ? stmt->body : stmt->alternative) Visit(child);
    }
  }

 public:
  CaseDispatchCost(const ExprPtr& block, int64_t value) : block_(block), value_(value) {}
  std::optional<size_t> Measure(const ControlForest& body) {
    for (const auto& stmt : body) Visit(stmt);
    return valid_ ? std::optional<size_t>(cost_) : std::nullopt;
  }
};

// Two disjoint intervals are the initial supported union subclass. This is a
// constant-size decision problem, independent of launch size and trip counts;
// it does not perform an all-pairs sequence search. More elaborate unions keep
// the balanced sequence tree. In particular, a gap is never replaced by a hull.
ControlForest ShareDisjointSequence(const ExprPtr& block, const Intervals& possible,
                                    const std::vector<CaseCommands>& cases, const ControlForest& incumbent,
                                    const std::set<const Var*>& uniform_variables,
                                    PhysicalSyncRealization& work, const Span& span) {
  if (cases.size() != 2 || !block || !SameControl(cases[0].body, cases[1].body)) return incumbent;
  UniformCommandBody uniform(uniform_variables);
  VisitControlExpressions(cases[0].body, uniform);
  if (!uniform.valid) return incumbent;
  const auto context = std::make_pair(possible.front().first, possible.back().second);
  std::vector<ExprPtr> predicates;
  std::set<int64_t> boundaries{context.first, context.second + 1};
  for (const auto& item : cases) {
    const auto lo = std::max(context.first, item.blocks.first);
    const auto hi = std::min(context.second, item.blocks.second);
    if (lo > hi) return incumbent;
    predicates.push_back(SyncInvocationGuard(block, context, {lo, hi}, span));
    boundaries.insert(lo);
    boundaries.insert(hi + 1);
  }
  if (!predicates[0] || !predicates[1]) return incumbent;
  const auto predicate = std::make_shared<Or>(predicates[0], predicates[1], DataType::BOOL, span);
  ControlForest candidate{ControlBranch(predicate, cases[0].body, std::nullopt, span, work)};
  const CaseCodeSize old_size(incumbent), new_size(candidate);
  work.case_sequence_visits += old_size.statements + new_size.statements;
  work.control_expression_visits += old_size.expressions + new_size.expressions;
  if (!new_size.Improves(old_size)) return incumbent;
  std::array<size_t, 2> old_max{}, new_max{};
  // Predicates are constant on these atoms. Intersecting possible intervals is
  // exact even when an atom begins in a proved-absent part of the launch domain.
  for (auto it = boundaries.begin(), next = std::next(it); next != boundaries.end(); ++it, ++next) {
    if (!DomainIntersects(possible, *it, *next - 1)) continue;
    const bool active = (*it >= cases[0].blocks.first && *it <= cases[0].blocks.second) ||
                        (*it >= cases[1].blocks.first && *it <= cases[1].blocks.second);
    const auto before = CaseDispatchCost(block, *it).Measure(incumbent);
    const auto after = CaseDispatchCost(block, *it).Measure(candidate);
    if (!before || !after) return incumbent;
    old_max[active] = std::max(old_max[active], *before);
    new_max[active] = std::max(new_max[active], *after);
  }
  if (new_max[0] > old_max[0] || new_max[1] > old_max[1]) return incumbent;
  ++work.shared_disjoint_case_cuts;
  return candidate;
}

// Reuse the same union rule after stripping an ordered common prefix/suffix
// from three exhaustive cases A/B/A. The middle residual must be empty. Thus
// this is still a two-interval union, not symbolic family or phase unification.
ControlForest ShareResidualSequence(const ExprPtr& block, const Intervals& possible,
                                    const std::vector<CaseCommands>& cases, const ControlForest& incumbent,
                                    const std::set<const Var*>& uniform_variables,
                                    PhysicalSyncRealization& work, const Span& span) {
  if (cases.size() != 3 || !block) return incumbent;
  Intervals covered;
  std::array<ControlForest, 3> bodies;
  UniformCommandBody uniform(uniform_variables);
  for (size_t i = 0; i < 3; ++i) {
    covered.push_back(cases[i].blocks);
    VisitControlExpressions(cases[i].body, uniform);
    FlattenControlSequence(cases[i].body, bodies[i], work);
  }
  // Prefix/suffix work also enters the inner maximum being cancelled, so its
  // valuation domain must be uniform. Coverage is required before hoisting it.
  if (!uniform.valid || !SubtractIntervals(possible, covered).empty()) return incumbent;
  size_t prefix = 0, suffix = 0;
  auto same = [&](size_t i, bool back) {
    if (prefix + suffix >= bodies[i].size()) return false;
    ++work.case_sequence_comparisons;
    return SameControl(bodies[0][back ? bodies[0].size() - 1 - suffix : prefix],
                       bodies[i][back ? bodies[i].size() - 1 - suffix : prefix]);
  };
  while (prefix + suffix < bodies[0].size() && same(1, false) && same(2, false)) ++prefix;
  while (prefix + suffix < bodies[0].size() && same(1, true) && same(2, true)) ++suffix;
  if (prefix + suffix != bodies[1].size() || prefix + suffix >= bodies[0].size() ||
      prefix + suffix >= bodies[2].size())
    return incumbent;
  std::vector<CaseCommands> residual;
  for (size_t i : {size_t{0}, size_t{2}})
    residual.push_back({cases[i].blocks,
                        ControlSequence({bodies[i].begin() + prefix, bodies[i].end() - suffix}, span, work)});
  const auto previous = work.shared_disjoint_case_cuts;
  // Certify against the actual selected incumbent, not a newly reconstructed
  // tree. The omitted shared prefix/suffix contain no launch-dependent work,
  // so reattaching them leaves this dispatch comparison exactly unchanged.
  auto middle = ShareDisjointSequence(block, possible, residual, incumbent, uniform_variables, work, span);
  if (work.shared_disjoint_case_cuts == previous) return incumbent;
  ControlForest candidate(bodies[0].begin(), bodies[0].begin() + prefix);
  candidate.insert(candidate.end(), middle.begin(), middle.end());
  candidate.insert(candidate.end(), bodies[0].end() - suffix, bodies[0].end());
  const CaseCodeSize old_size(incumbent), new_size(candidate);
  work.case_sequence_visits += old_size.statements + new_size.statements;
  work.control_expression_visits += old_size.expressions + new_size.expressions;
  if (!new_size.Improves(old_size)) {
    work.shared_disjoint_case_cuts = previous;
    return incumbent;
  }
  return candidate;
}

ControlForest ScheduleCommandCases(const ExprPtr& block, const Intervals& possible,
                                   const std::vector<CaseCommands>& cases, PhysicalSyncRealization& work,
                                   const Span& span, const std::set<const Var*>& uniform_variables) {
  ControlForest target;
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
  const auto coalesced = CoalesceCutCases(cases, possible, work);
  for (const auto& item : coalesced) {
    const auto selected = std::make_pair(std::max(context.first, item.blocks.first),
                                         std::min(context.second, item.blocks.second));
    if (selected.first > selected.second) continue;
    const auto guard = SyncInvocationGuard(block, context, selected, span);
    target.push_back(guard ? ControlBranch(guard, item.body, std::nullopt, span, work) : item.body);
  }
  if (coalesced.size() > 1) {
    auto candidate = CaseSequenceTree(block, span, work).Build(coalesced, possible);
    const CaseCodeSize old_size(target), new_size(candidate);
    work.case_sequence_visits += old_size.statements + new_size.statements;
    work.control_expression_visits += old_size.expressions + new_size.expressions;
    if (new_size.Improves(old_size)) {
      ++work.factored_case_cuts;
      target = std::move(candidate);
    }
  }
  target = ShareDisjointSequence(block, possible, coalesced, target, uniform_variables, work, span);
  return ShareResidualSequence(block, possible, coalesced, target, uniform_variables, work, span);
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
        AppendCase(retained->before_cases, blocks, commands.before, site.original->span_, result_);
        AppendCase(retained->after_cases, blocks, commands.after, site.original->span_, result_);
      }
    }
  }

 private:
  PhysicalSyncRealization& result_;
  std::unique_ptr<SyncTokenNames> names_;
  std::map<SyncSiteKey, std::vector<size_t>> incoming_, outgoing_;
  std::map<size_t, std::shared_ptr<const SyncEndpointFrame>> endpoint_frames_;
  using EndpointKey = std::pair<size_t, std::vector<std::pair<int64_t, int64_t>>>;
  std::map<EndpointKey, std::shared_ptr<const SyncEndpointDomain>> endpoint_domains_;
  std::map<const SyncEndpointDomain*, std::vector<ExprPtr>> endpoint_guards_;
  std::map<size_t, std::unique_ptr<CommandConditions>> conditions_;
  std::map<size_t, ExprPtr> ordinals_;
  CommandConditions& Conditions(size_t region) {
    auto& value = conditions_[region];
    if (!value) value = std::make_unique<CommandConditions>(input_.segments[region]);
    return *value;
  }
  std::shared_ptr<const SyncEndpointDomain> EndpointDomain(const Family& f, bool publish) {
    const auto r = publish ? f.source_region : f.target_region;
    auto retained = publish ? f.source_domain : f.target_domain;
    const auto& frames = input_.segments[r].nested_frames;
    if (retained) {
      const auto& context = *retained->frame;
      INTERNAL_CHECK(context.owner == input_.owner && context.blocks == input_.block_bounds &&
                     context.region == r && context.loops.size() == frames.size() &&
                     retained->ranges.size() == frames.size())
          << "Foreign endpoint domain";
      for (size_t i = 0; i < frames.size(); ++i)
        INTERNAL_CHECK(context.loops[i].loop == frames[i].loop && context.loops[i].count == frames[i].count &&
                       context.loops[i].start == frames[i].start && context.loops[i].step == frames[i].step &&
                       context.loops[i].start_expression == frames[i].start_expression)
            << "Incompatible original endpoint frame";
      return retained;
    }
    // Mathematics-only clients use the same domain representation. Decode once
    // at this compatibility boundary; actual-token clients never take this path.
    auto base = publish ? f.source_first : f.target_first;
    std::vector<std::pair<int64_t, int64_t>> ranges(frames.size());
    for (size_t i = frames.size(); i-- > 0;) {
      INTERNAL_CHECK(frames[i].count > 0) << "Physical endpoint in absent frame";
      const auto first = base % frames[i].count;
      base /= frames[i].count;
      const auto count = i < f.axes.size() ? f.axes[i].count : 1;
      INTERNAL_CHECK(i >= f.axes.size() || f.axes[i].frame == frames[i].loop) << "Physical scope identity";
      INTERNAL_CHECK(count > 0 && count <= frames[i].count - first) << "Physical endpoint range";
      ranges[i] = {first, first + count};
    }
    INTERNAL_CHECK(!base && f.axes.size() <= frames.size()) << "Physical scope arity";
    EndpointKey key{r, std::move(ranges)};
    if (auto found = endpoint_domains_.find(key); found != endpoint_domains_.end()) {
      return found->second;
    }
    auto& frame = endpoint_frames_[r];
    if (!frame)
      frame = std::make_shared<SyncEndpointFrame>(
          SyncEndpointFrame{input_.owner, input_.block_bounds, r, frames});
    auto domain = std::make_shared<SyncEndpointDomain>(SyncEndpointDomain{frame, key.second});
    endpoint_domains_.emplace(std::move(key), domain);
    return domain;
  }

  ExprPtr Integer(int64_t n, const Span& span) {
    return std::make_shared<ConstInt>(n, DataType::INDEX, span);
  }
  ExprPtr Ordinal(size_t region, const Span& span) {
    auto& value = ordinals_[region];
    if (!value) value = BuildOrdinal(region, span);
    return value;
  }
  ExprPtr BuildOrdinal(size_t region, const Span& span) {
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
  // Rewrite only on the original positive-step lattice. In addition to each
  // threshold, prove the normalized subtraction itself cannot wrap: endpoints
  // can both fit int64 while their difference does not. A symbolic start keeps
  // the established quotient expression; no new range assumption is introduced.
  std::optional<int64_t> OriginalBoundary(const SyncLoopFrame& frame, int64_t digit) const {
    const auto start = As<ConstInt>(frame.loop->start_), step = As<ConstInt>(frame.loop->step_);
    if (!start || !step || frame.start_expression || frame.step <= 0 || frame.count <= 0 ||
        start->value_ != frame.start || step->value_ != frame.step || digit < 0 || digit >= frame.count)
      return std::nullopt;
    const auto extent = __int128(frame.count - 1) * frame.step;
    const auto last = __int128(frame.start) + extent;
    const auto bound = __int128(frame.start) + __int128(digit) * frame.step;
    if (extent > std::numeric_limits<int64_t>::max() || last > std::numeric_limits<int64_t>::max() ||
        last < std::numeric_limits<int64_t>::min() || bound > std::numeric_limits<int64_t>::max() ||
        bound < std::numeric_limits<int64_t>::min())
      return std::nullopt;
    return static_cast<int64_t>(bound);
  }
  const std::vector<ExprPtr>& ScopedGuards(const Family& family, bool publish, const Span& span) {
    const auto domain = EndpointDomain(family, publish);
    if (auto found = endpoint_guards_.find(domain.get()); found != endpoint_guards_.end())
      return found->second;
    const auto& frames = domain->frame->loops;
    result_.endpoint_domain_entries += domain->ranges.size();
    std::vector<ExprPtr> guards;
    for (size_t i = 0; i < frames.size(); ++i) {
      const auto [first, end] = domain->ranges[i];
      INTERNAL_CHECK(first >= 0 && end > first && end <= frames[i].count) << "Invalid endpoint interval";
      if (!first && end == frames[i].count) continue;
      // Each requested threshold is within the admitted frame, including
      // nondivisible stop values. Never substitute stop-step for the last IV.
      const auto first_bound = OriginalBoundary(frames[i], first);
      const auto end_bound = end < frames[i].count ? OriginalBoundary(frames[i], end) : first_bound;
      const bool original = first_bound && end_bound;
      const auto value = original ? frames[i].loop->loop_var_ : FrameDigit(frames[i], span);
      auto boundary = [&](int64_t digit) {
        if (original) {
          ++result_.normalized_endpoint_bounds;
          return Integer(*OriginalBoundary(frames[i], digit), span);
        }
        return Integer(digit, span);
      };
      if (end - first == 1) {
        guards.push_back(std::make_shared<Eq>(value, boundary(first), DataType::BOOL, span));
      } else {
        if (first) guards.push_back(std::make_shared<Ge>(value, boundary(first), DataType::BOOL, span));
        if (end < frames[i].count)
          guards.push_back(std::make_shared<Lt>(value, boundary(end), DataType::BOOL, span));
      }
    }
    result_.endpoint_predicates_constructed += guards.size();
    for (auto& guard : guards) guard = Conditions(domain->frame->region).Simplify(guard);
    return endpoint_guards_.emplace(domain.get(), std::move(guards)).first->second;
  }
  ControlPtr Guard(ExprPtr condition, ControlPtr body) {
    ++result_.guard_sites;
    if (auto known = As<ConstBool>(condition)) {
      return known->value_ ? body : ControlSequence({}, body->span, result_);
    }
    return ControlBranch(condition, body, std::nullopt, body->span, result_);
  }
  ControlPtr Command(const Family& f, bool publish, const Span& span) {
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
    ControlForest commands;
    if (src == dst) {
      INTERNAL_CHECK(dst == PipeType::M || dst == PipeType::V)
          << "Only qualified M/V barriers may be emitted; scalar barriers are forbidden";
      auto call = OpRegistry::GetInstance().Create(dst == PipeType::M ? "system.bar_m" : "system.bar_v", {},
                                                   {}, span);
      commands.push_back(ControlCommand(call, span, result_));
      ++result_.barrier_sites;
    } else {
      ExprPtr residue;
      if (f.ids.size() > 1)
        residue = std::make_shared<FloorMod>(index, Integer(f.ids.size(), span), DataType::INDEX, span);
      ControlPtr selection;
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
        ControlPtr command = ControlCommand(call, span, result_);
        if (residue) {
          ++result_.guard_sites;
          command = ControlBranch(
              Conditions(r).Simplify(std::make_shared<Eq>(residue, Integer(i, span), DataType::BOOL, span)),
              command, selection ? std::optional<ControlPtr>(selection) : std::nullopt, span, result_);
        }
        selection = command;
        if (publish)
          ++result_.set_sites;
        else
          ++result_.wait_sites;
      }
      if (selection) commands.push_back(selection);
    }
    ControlPtr body = ControlSequence(commands, span, result_);
    // The range guard surrounds residue arithmetic: no absent predecessor is
    // evaluated/consumed. Final-only exports never publish on earlier visits.
    if (scoped) {
      const auto& guards = ScopedGuards(f, publish, span);
      for (auto guard = guards.rbegin(); guard != guards.rend(); ++guard) body = Guard(*guard, body);
      return body;
    }
    if (first + f.count < input_.segments[r].count)
      body = Guard(Conditions(r).Simplify(
                       std::make_shared<Lt>(ordinal, Integer(first + f.count, span), DataType::BOOL, span)),
                   body);
    if (first)
      body = Guard(
          Conditions(r).Simplify(std::make_shared<Ge>(ordinal, Integer(first, span), DataType::BOOL, span)),
          body);
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
    const auto remaining = CaseCodeSize(cuts.before).guards + CaseCodeSize(cuts.after).guards;
    INTERNAL_CHECK(result_.guard_sites >= previous_guards + remaining) << "Unexpected new command guards";
    result_.simplified_guard_sites += result_.guard_sites - previous_guards - remaining;
    result_.guard_sites = previous_guards + remaining;
    cuts.before = CommandGuards().Build(cuts.before, result_);
    cuts.after = CommandGuards().Build(cuts.after, result_);
    return cuts;
  }
  StmtPtr Wrap(const StmtPtr& payload, SyncSiteKey site) override {
    auto cuts = Commands(site, payload->span_);
    if (!names_) names_ = std::make_unique<SyncTokenNames>(input_.owner);
    auto before = MaterializeCommandValues(MaterializeControl(cuts.before, result_), *names_, result_);
    const auto after = MaterializeCommandValues(MaterializeControl(cuts.after, result_), *names_, result_);
    before.push_back(payload);
    before.insert(before.end(), after.begin(), after.end());
    return std::make_shared<SeqStmts>(before, payload->span_);
  }
};
// The command builder above owns all protocol mechanics. This adapter only
// installs already-qualified case cuts in the unchanged original payload tree.
class InvocationEmitter : public IRMutator {
  const CutInventory& inventory_;
  SyncTokenNames names_;
  PhysicalSyncRealization& work_;
  StmtPtr Wrap(const StmtPtr& payload, const PhysicalCuts& cuts) {
    auto statements = MaterializeCommandValues(MaterializeControl(cuts.before, work_), names_, work_);
    statements.push_back(payload);
    const auto after = MaterializeCommandValues(MaterializeControl(cuts.after, work_), names_, work_);
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
                                          PhysicalSyncRealization& work, const Span& span,
                                          const std::set<const Var*>& uniform_variables) {
  std::vector<CaseCommands> descriptors;
  for (const auto& item : cases) descriptors.push_back({item.blocks, ImportControl(item.body, work)});
  return MaterializeControl(ScheduleCommandCases(block, possible, descriptors, work, span, uniform_variables),
                            work);
}
std::vector<StmtPtr> ShareSyncCommandGuards(const std::vector<StmtPtr>& commands,
                                            PhysicalSyncRealization& counters) {
  ControlForest descriptors;
  for (const auto& command : commands) descriptors.push_back(ImportControl(command, counters));
  return MaterializeControl(CommandGuards().Build(descriptors, counters), counters);
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
    auto emit = [&](const std::vector<CaseCommands>& source, ControlForest& target) {
      target = ScheduleCommandCases(block, cut->domain, source, control_work, owner->span_,
                                    cut->uniform_variables);
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
