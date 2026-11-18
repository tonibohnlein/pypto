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
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {
// Freshen only logical completion identities through the parent's naming owner.
// Payload variables, original induction variables and storage descriptors retain
// their original identities; cloning all definitions would lose region frames.
class FreshChildTokens : public IRMutator {
 public:
  explicit FreshChildTokens(SyncTokenNames& names) : names_(names) {}

 protected:
  ExprPtr VisitExpr_(const VarPtr& variable) override {
    const auto found = variables_.find(variable.get());
    return found == variables_.end() ? variable : found->second;
  }
  ExprPtr VisitExpr_(const IterArgPtr& variable) override { return VisitExpr_(VarPtr(variable)); }
  StmtPtr VisitStmt_(const AssignStmtPtr& assignment) override {
    const auto value = VisitExpr(assignment->value_);
    const auto variable = Fresh(assignment->var_);
    return std::make_shared<AssignStmt>(variable, value, assignment->span_, assignment->leading_comments_);
  }
  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    std::vector<IterArgPtr> arguments;
    for (const auto& argument : loop->iter_args_) {
      CHECK(As<SyncTokenType>(argument->GetType()))
          << "Eventful child scalar carries need original-frame binding evidence";
      auto fresh = std::make_shared<IterArg>(names_.Next("sync_child_completion_"), argument->GetType(),
                                             VisitExpr(argument->initValue_), argument->span_);
      variables_.emplace(argument.get(), fresh);
      arguments.push_back(fresh);
    }
    std::vector<VarPtr> results;
    for (const auto& variable : loop->return_vars_) results.push_back(Fresh(variable));
    return std::make_shared<ForStmt>(loop->loop_var_, loop->start_, loop->stop_, loop->step_, arguments,
                                     VisitStmt(loop->body_), results, loop->span_, loop->kind_, loop->attrs_,
                                     loop->leading_comments_, loop->hardware_loop_policy_);
  }
  StmtPtr VisitStmt_(const IfStmtPtr& branch) override {
    std::vector<VarPtr> results;
    for (const auto& variable : branch->return_vars_) results.push_back(Fresh(variable));
    const auto other =
        branch->else_body_ ? std::optional<StmtPtr>{VisitStmt(*branch->else_body_)} : std::nullopt;
    return std::make_shared<IfStmt>(branch->condition_, VisitStmt(branch->then_body_), other, results,
                                    branch->span_, branch->leading_comments_);
  }

 private:
  SyncTokenNames& names_;
  std::unordered_map<const Var*, VarPtr> variables_;
  VarPtr Fresh(const VarPtr& variable) {
    if (!As<SyncTokenType>(variable->GetType())) return variable;
    const auto found = variables_.find(variable.get());
    if (found != variables_.end()) return found->second;
    auto fresh =
        std::make_shared<Var>(names_.Next("sync_child_completion_"), variable->GetType(), variable->span_);
    variables_.emplace(variable.get(), fresh);
    return fresh;
  }
};

class Insert : public IRMutator {
 public:
  explicit Insert(const ConservativeSyncAnalysis& analysis) : names_(analysis.Function()) {
    // Build the full replacement inventory first. Every service uses its
    // consumer's CURRENT control and cut; no historical guard or private state
    // is introduced. Reusing the same static statement in two contexts requires
    // distinct original cut evidence and is refused rather than conflated.
    for (const auto& site : analysis.Sites()) {
      const auto destination = site.operation.Implementation().Pipe();
      std::vector<std::pair<PipeType, PipeType>> services;
      for (const auto source : site.source_pipes) {
        services.emplace_back(source, destination);
      }
      Wrap(site.cut, services);
    }
    for (const auto& entry : analysis.ChildEntries()) {
      if (entry.inline_subblock) {
        const auto& block = *entry.inline_subblock;
        const auto first_stmt = entry.window_seq->stmts_[entry.window_begin];
        std::vector<StmtPtr> statements;
        for (const auto& [source, destination] : entry.services) {
          const auto forward = Record(source, statements, first_stmt->span_);
          Wait(forward, destination, statements, first_stmt->span_);
        }
        const auto fresh_spliced = FreshChildTokens(names_).VisitStmt(block.spliced_seq);
        if (const auto sseq = As<SeqStmts>(fresh_spliced)) {
          statements.insert(statements.end(), sseq->stmts_.begin(), sseq->stmts_.end());
        }
        CHECK(replacements_
                  .emplace(first_stmt.get(),
                           std::make_shared<SeqStmts>(std::move(statements), first_stmt->span_))
                  .second)
            << "Conservative prefix entries require distinct original static cuts";
        for (size_t k = entry.window_begin + 1; k < entry.window_end; ++k) {
          const auto& drop = entry.window_seq->stmts_[k];
          CHECK(replacements_
                    .emplace(drop.get(), std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, drop->span_))
                    .second)
              << "Conservative prefix sub-block statements require distinct original cuts";
        }
        continue;
      }

      auto recipe = entry.eventful ? FreshChildTokens(names_).VisitStmt(entry.eventful->LogicalRecipe())
                                   : StmtPtr(entry.scope);
      if (entry.first_visit && entry.eventful) {
        if (const auto marked = As<ForStmt>(recipe); marked && marked->HasAttr(kClosedSyncChildAttr)) {
          auto updated = std::make_shared<ForStmt>(*marked);
          updated->attrs_.emplace_back(kClosedSyncChildFirstVisitAttr, true);
          recipe = std::move(updated);
        } else if (const auto seq = As<SeqStmts>(recipe); seq && !seq->stmts_.empty()) {
          if (const auto marked_back = As<ForStmt>(seq->stmts_.back());
              marked_back && marked_back->HasAttr(kClosedSyncChildAttr)) {
            auto updated = std::make_shared<ForStmt>(*marked_back);
            updated->attrs_.emplace_back(kClosedSyncChildFirstVisitAttr, true);
            auto stmts = seq->stmts_;
            stmts.back() = std::move(updated);
            recipe = std::make_shared<SeqStmts>(std::move(stmts), seq->span_);
          }
        }
      }
      if (entry.IsSubblockSlice()) {
        const auto first_stmt = entry.window_seq->stmts_[entry.window_begin];
        Wrap(first_stmt, entry.services, recipe);
        for (size_t k = entry.window_begin + 1; k < entry.window_end; ++k) {
          const auto& drop = entry.window_seq->stmts_[k];
          CHECK(replacements_
                    .emplace(drop.get(), std::make_shared<SeqStmts>(std::vector<StmtPtr>{}, drop->span_))
                    .second)
              << "Conservative prefix sub-block statements require distinct original cuts";
        }
      } else {
        Wrap(entry.scope, entry.services, recipe);
      }
    }
  }
  StmtPtr VisitStmt(const StmtPtr& statement) override {
    if (auto replacement = replacements_.find(statement.get()); replacement != replacements_.end())
      return replacement->second;
    return IRMutator::VisitStmt(statement);
  }

 protected:
  // Compound-statement mutators dispatch direct arm/body statements through
  // StmtFunctor. Handle leaves here as well as through generic VisitStmt, so a
  // one-operation If arm receives exactly the same services as a Seq body.
  StmtPtr VisitStmt_(const EvalStmtPtr& statement) override {
    const auto replacement = replacements_.find(statement.get());
    return replacement == replacements_.end() ? StmtPtr(statement) : replacement->second;
  }
  StmtPtr VisitStmt_(const AssignStmtPtr& statement) override {
    const auto replacement = replacements_.find(statement.get());
    return replacement == replacements_.end() ? StmtPtr(statement) : replacement->second;
  }
  StmtPtr VisitStmt_(const ForStmtPtr& loop) override {
    const auto replacement = replacements_.find(loop.get());
    return replacement == replacements_.end() ? IRMutator::VisitStmt_(loop) : replacement->second;
  }
  StmtPtr VisitStmt_(const SeqStmtsPtr& sequence) override {
    return RewriteSyncSequence(sequence, [this](const StmtPtr& statement) { return VisitStmt(statement); });
  }
  ExprPtr VisitExpr(const ExprPtr& expression) override { return expression; }

 private:
  SyncTokenNames names_;
  std::map<const Stmt*, StmtPtr> replacements_;
  void Wrap(const StmtPtr& cut, const std::vector<std::pair<PipeType, PipeType>>& services,
            StmtPtr retained = nullptr) {
    std::vector<StmtPtr> statements;
    if (const auto recipe = As<SeqStmts>(retained)) {
      CHECK(!recipe->stmts_.empty()) << "Eventful child recipe has no whole invocation";
      statements.insert(statements.end(), recipe->stmts_.begin(), recipe->stmts_.end() - 1);
      retained = recipe->stmts_.back();
    }
    for (const auto& [source, destination] : services) {
      const auto forward = Record(source, statements, cut->span_);
      Wait(forward, destination, statements, cut->span_);
    }
    statements.push_back(retained ? std::move(retained) : cut);
    CHECK(replacements_.emplace(cut.get(), std::make_shared<SeqStmts>(statements, cut->span_)).second)
        << "Conservative prefix entries require distinct original static cuts";
  }
  VarPtr Record(PipeType pipe, std::vector<StmtPtr>& output, const Span& span) {
    const auto operation =
        OpRegistry::GetInstance().Create("sync.record", {}, {{"pipe", static_cast<int>(pipe)}}, span);
    const auto variable =
        std::make_shared<Var>(names_.Next("sync_parent_completion_"), operation->GetType(), span);
    output.push_back(std::make_shared<AssignStmt>(variable, operation, span));
    return variable;
  }
  static void Wait(const VarPtr& token, PipeType destination, std::vector<StmtPtr>& output,
                   const Span& span) {
    const auto operation = OpRegistry::GetInstance().Create(
        "sync.wait", {token}, {{"dst_pipe", static_cast<int>(destination)}}, span);
    output.push_back(std::make_shared<EvalStmt>(operation, span));
  }
};
}  // namespace

FunctionPtr InsertConservativePrefixes(const ConservativeSyncAnalysis& analysis) {
  const auto result = Insert(analysis).VisitFunction(analysis.Function());
  CheckLogicalSync(result);
  auto reported = std::make_shared<Function>(*result);
  reported->attrs_.emplace_back(kAttrAutosyncAnalysisPolicy, std::string("conservative_prefix"));
  reported->attrs_.emplace_back(kAttrAutosyncAccessPrecision, std::string("qualified_storage_enclosures"));
  reported->attrs_.emplace_back(kAttrAutosyncRealizationOrder, std::string("preserves_selected_token_order"));
  // Reporting only: the immutable analysis owns the actual mathematics and
  // original scopes. LowerSync never treats these counters as certificates.
  const auto& selection = analysis.ChildSelection();
  reported->attrs_.emplace_back("autosync_child_attempt_count", static_cast<int>(selection.attempts.size()));
  const auto retained_child_count = static_cast<int>(std::count_if(
      analysis.ChildEntries().begin(), analysis.ChildEntries().end(),
      [](const ConservativeSyncChildEntry& entry) { return !entry.inline_subblock.has_value(); }));
  reported->attrs_.emplace_back("autosync_retained_child_count", retained_child_count);
  std::string refusals;
  for (const auto& attempt : selection.attempts) {
    if (attempt.child) continue;
    if (!refusals.empty()) refusals += "\n";
    refusals += attempt.scope->loop_var_->name_hint_ + ": " + EventfulSyncChildStageName(attempt.stage) +
                ": " + attempt.reason;
  }
  if (!refusals.empty()) reported->attrs_.emplace_back("autosync_child_refusals", refusals);
  return reported;
}
}  // namespace pypto::ir::sync
