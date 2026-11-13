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

#include "src/ir/transforms/utils/physical_sync_commands.h"

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pypto/core/logging.h"
#include "pypto/ir/scalar_expr.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/transforms/base/visitor.h"
#include "pypto/ir/transforms/structural_comparison.h"
#include "src/ir/transforms/utils/physical_sync_allocation.h"
#include "src/ir/transforms/utils/sync_invocation_domains.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace physical {
// Equality is optional optimization evidence. An unsupported comparison must
// preserve the accepted unfactored plan, rather than become an export refusal.
bool SameControl(const IRNodePtr& left, const IRNodePtr& right) {
  try {
    return structural_equal(left, right, false);
  } catch (const StructuralComparisonUnsupported&) {
    return false;
  }
}

// Scope-local value numbering for generated command predicates. Count only
// strict, pure expression trees in conditions at this sequence level. Insert
// a definition at its FIRST evaluation, never at a loop preheader or outside
// a guard. Children have independent tables; no branch-private value escapes.
// Hash collisions lose sharing rather than inducing an all-pairs search.
bool CommandValueCandidate(const ExprPtr& expression) {
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
bool PureCommandExpression(const ExprPtr& expression, size_t* inspections) {
  ++*inspections;
  const auto type = As<ScalarType>(expression->GetType());
  if (!type || (type->dtype_ != DataType::INDEX && type->dtype_ != DataType::BOOL)) return false;
  if (As<Var>(expression) || As<ConstInt>(expression) || As<ConstBool>(expression)) return true;
  const auto binary = As<BinaryExpr>(expression);
  if (!binary || As<And>(expression) || As<Or>(expression)) return false;
  if (As<FloorDiv>(expression) || As<FloorMod>(expression)) {
    auto divisor = As<ConstInt>(binary->right_);
    if (!divisor || divisor->value_ <= 0) return false;
  }
  return PureCommandExpression(binary->left_, inspections) &&
         PureCommandExpression(binary->right_, inspections);
}

class CommandValues : public IRMutator {
  SyncTokenNames& names_;
  PhysicalSyncRealization& work_;
  struct Entry {
    ExprPtr expression;
    size_t uses = 0;
    VarPtr value;
  };
  using Table = std::map<uint64_t, Entry>;

 private:
  bool Pure(const ExprPtr& expression) {
    return PureCommandExpression(expression, &work_.control_expression_visits);
  }
  void Count(const ExprPtr& expression, Table& table) {
    ++work_.control_expression_visits;
    const auto binary = As<BinaryExpr>(expression);
    if (!binary) return;
    if (CommandValueCandidate(expression)) {
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
      auto found =
          CommandValueCandidate(expression) ? table_.find(structural_hash(expression, false)) : table_.end();
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

// Share adjacent decisions directly in descriptors. For an immutable condition
// c, (if c A else B); (if c C else D) is (if c A;C else B;D).
// Append each supplied arm once and freeze once: never copy a growing prefix.
// Empty arms and duplicate command leaves retain their original meaning. This
// operates only within one command forest; payloads and definitions are fences.
class CommandGuards {
  struct Node {
    ExprPtr condition;
    ControlPtr leaf;
    std::vector<Node> children, alternative;
  };
  std::vector<Node> roots_;
  size_t shared_ = 0, visits_ = 0, expressions_ = 0;
  void Append(std::vector<Node>& nodes, const ControlPtr& statement) {
    ++visits_;
    if (!statement->condition && !statement->command) {
      for (const auto& child : statement->body) Append(nodes, child);
    } else if (statement->condition && (statement->alternative.empty() ||
                                        PureCommandExpression(statement->condition, &expressions_))) {
      if (nodes.empty() || !nodes.back().condition ||
          !SameControl(nodes.back().condition, statement->condition))
        nodes.push_back({statement->condition, nullptr, {}, {}});
      else
        ++shared_;
      for (const auto& child : statement->body) Append(nodes.back().children, child);
      for (const auto& child : statement->alternative) Append(nodes.back().alternative, child);
    } else {
      nodes.push_back({nullptr, statement, {}, {}});
    }
  }
  static ControlForest Freeze(const std::vector<Node>& nodes, PhysicalSyncRealization& work) {
    ControlForest result;
    for (const auto& node : nodes) {
      if (node.condition) {
        std::optional<ControlPtr> alternative;
        if (!node.alternative.empty())
          alternative = ControlSequence(Freeze(node.alternative, work), node.condition->span_, work);
        result.push_back(ControlBranch(
            node.condition, ControlSequence(Freeze(node.children, work), node.condition->span_, work),
            alternative, node.condition->span_, work));
      } else {
        result.push_back(node.leaf);
      }
    }
    return result;
  }

 public:
  ControlForest Build(const ControlForest& commands, PhysicalSyncRealization& result, bool endpoint = true) {
    for (const auto& command : commands) Append(roots_, command);
    if (endpoint) {
      INTERNAL_CHECK(result.guard_sites >= shared_) << "Internal error: command guard accounting";
      result.guard_sites -= shared_;
      result.shared_guard_sites += shared_;
    } else {
      // Case decisions were constructed after endpoint accounting. Do not
      // subtract them from the endpoint-only inventory.
      result.shared_composed_guard_sites += shared_;
    }
    result.control_expression_visits += expressions_;
    result.control_statement_visits += visits_;
    // Composed forests are already canonical sequences. Reuse their immutable
    // descriptors when inspection found no sharing; do not reconstruct them.
    if (!endpoint && !shared_) return commands;
    return Freeze(roots_, result);
  }
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
class Expressions : public IRVisitor {
 public:
  size_t count = 0;
  void VisitExpr(const ExprPtr& expression) override {
    ++count;
    IRVisitor::VisitExpr(expression);
  }
};
void CaseCodeSize::Visit(const ControlPtr& node) {
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
CaseCodeSize::CaseCodeSize(const ControlForest& body) {
  for (const auto& node : body) Visit(node);
}
bool CaseCodeSize::Improves(const CaseCodeSize& old) const {
  return commands <= old.commands && guards <= old.guards && expressions <= old.expressions &&
         (commands < old.commands || guards < old.guards || expressions < old.expressions);
}

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
  target = ShareResidualSequence(block, possible, coalesced, target, uniform_variables, work, span);
  return CommandGuards().Build(target, work, false);
}

ControlForest ShareCommandGuards(const ControlForest& commands, PhysicalSyncRealization& work,
                                 bool endpoint) {
  return CommandGuards().Build(commands, work, endpoint);
}

}  // namespace physical

using namespace physical;
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

}  // namespace pypto::ir::sync
