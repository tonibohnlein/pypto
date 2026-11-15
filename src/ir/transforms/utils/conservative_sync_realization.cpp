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
#include <string>
#include <vector>

#include "pypto/ir/op_registry.h"
#include "pypto/ir/transforms/base/mutator.h"
#include "pypto/ir/verifier/sync.h"
#include "src/ir/transforms/utils/conservative_sync_dataflow.h"
#include "src/ir/transforms/utils/sync_statement_rewrite.h"
#include "src/ir/transforms/utils/sync_token_names.h"

namespace pypto::ir::sync {
namespace {
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
    for (const auto& entry : analysis.ChildEntries()) Wrap(entry.scope, entry.services);
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
  void Wrap(const StmtPtr& cut, const std::vector<std::pair<PipeType, PipeType>>& services) {
    std::vector<StmtPtr> statements;
    for (const auto& [source, destination] : services) {
      const auto forward = Record(source, statements, cut->span_);
      Wait(forward, destination, statements, cut->span_);
      if (source != destination) {
        // The selected logical program includes the acknowledgment order;
        // allocation must not introduce reverse dependencies as a repair.
        const auto returned = Record(destination, statements, cut->span_);
        Wait(returned, source, statements, cut->span_);
      }
    }
    statements.push_back(cut);
    CHECK(replacements_.emplace(cut.get(), std::make_shared<SeqStmts>(statements, cut->span_)).second)
        << "Conservative prefix entries require distinct original static cuts";
  }
  VarPtr Record(PipeType pipe, std::vector<StmtPtr>& output, const Span& span) {
    const auto operation =
        OpRegistry::GetInstance().Create("sync.record", {}, {{"pipe", static_cast<int>(pipe)}}, span);
    const auto variable = std::make_shared<Var>(names_.Next(), operation->GetType(), span);
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
  reported->attrs_.emplace_back(kAttrAutosyncAccessPrecision, std::string("whole_memory_space_enclosures"));
  reported->attrs_.emplace_back(kAttrAutosyncRealizationOrder, std::string("preserves_selected_token_order"));
  return reported;
}
}  // namespace pypto::ir::sync
